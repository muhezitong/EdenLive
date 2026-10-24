// SPDX-License-Identifier: Apache-2.0
/**
 * host/standalone.js — drive an unmodified standalone game page frame by frame.
 *
 * Games imported from the GamePrototypeTurbo skill own their loop: `start()`
 * schedules requestAnimationFrame and reads time from the RAF timestamp or
 * performance.now(). Rather than rewriting each game into the createRuntime()
 * contract, this classic script runs before any page module and replaces both
 * clocks with a virtual one. Nothing runs until the Python worker asks for a
 * tick, and every tick is exactly one RAF callback 1/60 s after the previous
 * one, so the game's own fixed-step loop advances in lockstep with Eden's
 * simulation timeline.
 *
 * It exposes the same window.gameBackend surface as host/backend.js.
 * State comes from the game's window.__gameTest.read(), the observation
 * contract every accepted skill game already satisfies.
 *
 * Real setTimeout/setInterval are left alone; a game that times gameplay with
 * them is outside the skill's clock contract and will drift.
 */
(() => {
  const SIM_HZ = 60;
  const FRAME_MS = 1000 / SIM_HZ;
  const config = window.gameStandalone || {};
  const restartKeys = Array.isArray(config.restartKeys) && config.restartKeys.length
    ? config.restartKeys : ['r'];
  const eventPrompts = config.eventPrompts || {};

  let virtualNow = 0;
  let queue = new Map();
  let nextHandle = 1;
  window.requestAnimationFrame = (callback) => {
    const handle = nextHandle++;
    queue.set(handle, callback);
    return handle;
  };
  window.cancelAnimationFrame = (handle) => { queue.delete(handle); };
  performance.now = () => virtualNow;

  // One RAF turn: callbacks queued before it run, callbacks they queue wait.
  function runFrame() {
    const due = queue;
    queue = new Map();
    for (const callback of due.values()) {
      try {
        callback(virtualNow);
      } catch (error) {
        console.error(error);
      }
    }
  }

  // The client forwards KeyboardEvent.key lowercased; rebuild key and code so
  // games reading either one see the press.
  const NAMED = {
    ' ': [' ', 'Space'], space: [' ', 'Space'], shift: ['Shift', 'ShiftLeft'],
    escape: ['Escape', 'Escape'], enter: ['Enter', 'Enter'], tab: ['Tab', 'Tab'],
    backspace: ['Backspace', 'Backspace'],
  };
  function keyAndCode(action) {
    const lower = String(action).toLowerCase();
    if (NAMED[lower]) return NAMED[lower];
    if (lower.length === 1 && /[a-z]/.test(lower)) return [lower, `Key${lower.toUpperCase()}`];
    if (lower.length === 1 && /[0-9]/.test(lower)) return [lower, `Digit${lower}`];
    if (lower.startsWith('arrow')) {
      const name = `Arrow${lower.slice(5, 6).toUpperCase()}${lower.slice(6)}`;
      return [name, name];
    }
    return [action, action];
  }

  function key(action, phase) {
    const [k, code] = keyAndCode(action);
    const target = document.activeElement || document.body || document;
    target.dispatchEvent(new KeyboardEvent(phase === 'down' ? 'keydown' : 'keyup', {
      key: k, code, bubbles: true, cancelable: true,
    }));
  }

  // EdenLive's server turns the browser's mouse move events into
  // {action: 'mouse-move', phase: 'move', dx, dy} payloads; dispatch them as
  // mousemove events so games that read movementX/Y can turn the camera.
  // dx/dy are fractions of the viewport; scaled to pixels so movementX/Y
  // keep their usual magnitude.
  function mouseMove(dx, dy) {
    const target = canvas || document.body;
    target.dispatchEvent(new MouseEvent('mousemove', {
      movementX: Math.round((+dx || 0) * window.innerWidth),
      movementY: Math.round((+dy || 0) * window.innerHeight),
      bubbles: true, cancelable: true,
    }));
  }

  function dispatchEvents(events) {
    for (const {action, phase, dx, dy} of events || []) {
      if (action === 'mouse-move') { mouseMove(dx, dy); continue; }
      // Overlay buttons send restart as a client action; the game only knows
      // its own restart key, so press that instead.
      if (action === 'restart' || action === 'reset') {
        if (phase !== 'down') continue;
        for (const restartKey of restartKeys) { key(restartKey, 'down'); key(restartKey, 'up'); }
        continue;
      }
      key(action, phase);
    }
  }

  function readState() {
    let snapshot;
    try {
      snapshot = window.__gameTest?.read?.();
    } catch {
      return null;
    }
    if (!snapshot || typeof snapshot !== 'object') return null;
    const {frame, clock, ...state} = snapshot;
    if (Array.isArray(state.events)) {
      state.events = state.events.map((event) => {
        const prompt = event && typeof event.prompt === 'string' ? event.prompt : eventPrompts[event?.type];
        return {...event, prompt: typeof prompt === 'string' ? prompt : null};
      });
    }
    return state;
  }

  let canvas = null;
  let scratch = null;
  // Like host/backend.js, the model sees only the scene canvas: DOM overlays
  // (HUD, key hints, result cards) are never captured.
  // A game that pins its own pixel ratio renders larger than the viewport;
  // resample so the fallback frame still matches the model's frame size.
  function canvasShot() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    if (canvas.width === width && canvas.height === height) {
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    scratch ??= document.createElement('canvas');
    scratch.width = width;
    scratch.height = height;
    scratch.getContext('2d').drawImage(canvas, 0, 0, width, height);
    return scratch.toDataURL('image/jpeg', 0.8);
  }

  let tick = 0;
  function simulateTo(targetTick) {
    if (!Number.isSafeInteger(targetTick) || targetTick < tick) throw Error('Invalid target tick');
    while (tick < targetTick) {
      tick++;
      virtualNow = tick * FRAME_MS;
      runFrame();
    }
  }

  // The game draws inside its RAF callback, so the drawing buffer still holds
  // that frame when this runs in the same task.
  function capture() {
    return {
      tick,
      width: window.innerWidth,
      height: window.innerHeight,
      ui: canvasShot(),
      state: readState(),
    };
  }

  const backend = {
    dispatch(events) { dispatchEvents(events); },
    advanceToFrame({events, targetTick}) {
      dispatchEvents(events);
      simulateTo(targetTick);
      return capture();
    },
    // Zero-length frame: input handlers and the restart edge run, but the
    // game's fixed-step clock sees rawDt 0 and does not simulate.
    probe(events) {
      dispatchEvents(events);
      runFrame();
      return {tick, state: readState()};
    },
    async advanceChunk({events, frames}) {
      dispatchEvents(events);
      const captured = [];
      for (const frame of frames) {
        simulateTo(frame.targetTick);
        captured.push(capture());
      }
      return {tick, frames: captured};
    },
    destroy() { queue.clear(); },
  };

  // start() may await module loads; publish the backend only once the game
  // has a canvas and an observation hook, so the worker never drives a page
  // that is still booting. The first frame paints the initial scene.
  const ready = () => {
    const fatal = document.getElementById('fatal');
    if (fatal && fatal.style.display === 'block' && fatal.textContent) {
      console.error(`Game failed to start: ${fatal.textContent}`);
      return;
    }
    canvas = document.querySelector('#app canvas') || document.querySelector('canvas');
    if (!canvas || !window.__gameTest) {
      setTimeout(ready, 20);
      return;
    }
    runFrame();
    window.gameBackend = backend;
  };
  setTimeout(ready, 0);
})();
