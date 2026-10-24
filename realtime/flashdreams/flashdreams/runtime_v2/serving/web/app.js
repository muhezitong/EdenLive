// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
// Modified by Vast AI for EdenLive.
// SPDX-License-Identifier: Apache-2.0

const iceServers = window.__ICE_SERVERS__ || [];
const peer = new RTCPeerConnection({
  iceServers,
  // Force TURN relay when a relay is configured so media does not stall on
  // host candidates that cannot reach the pod across the network.
  iceTransportPolicy: iceServers.length ? "relay" : "all",
});
let previousInboundSample = null;
// The workbench reads browser-reported transport stats while its play view is
// open. RTT is network round trip, not glass-to-glass video latency.
window.flashdreamsWebRTCStats = async () => {
  if (peer.connectionState !== "connected") {
    previousInboundSample = null;
    return {connected: false, rtt_ms: null, receive_fps: null};
  }
  const report = await peer.getStats();
  let rttMs = null;
  let inbound = null;
  let selectedPairId = null;
  report.forEach(stat => {
    if (stat.type === "transport" && stat.selectedCandidatePairId) {
      selectedPairId = stat.selectedCandidatePairId;
    }
  });
  report.forEach(stat => {
    if (stat.type === "candidate-pair" && stat.state === "succeeded" &&
        (stat.id === selectedPairId || (!selectedPairId && stat.nominated)) &&
        Number.isFinite(stat.currentRoundTripTime)) {
      rttMs = stat.currentRoundTripTime * 1000;
    }
    if (stat.type === "inbound-rtp" && !stat.isRemote &&
        (stat.kind === "video" || stat.mediaType === "video") &&
        (!inbound || stat.mid === "0")) {
      inbound = stat;
    }
  });
  let receiveFps = Number.isFinite(inbound?.framesPerSecond) ? inbound.framesPerSecond : null;
  if (receiveFps === null && Number.isFinite(inbound?.framesDecoded)) {
    const current = {id: inbound.id, frames: inbound.framesDecoded, timestamp: inbound.timestamp};
    if (previousInboundSample?.id === current.id && current.timestamp > previousInboundSample.timestamp) {
      receiveFps = Math.max(0, (current.frames - previousInboundSample.frames) * 1000 /
        (current.timestamp - previousInboundSample.timestamp));
    }
    previousInboundSample = current;
  }
  return {connected: true, rtt_ms: rttMs, receive_fps: receiveFps};
};
const controls = peer.createDataChannel("controls");
const pointerControls = peer.createDataChannel("pointer-controls");
peer.addTransceiver("video", {direction: "recvonly"});
if (window.__PAIRED_VIDEO__) peer.addTransceiver("video", {direction: "recvonly"});
const video = document.getElementById("video");
const overlayHost = document.getElementById("game-overlay");
const status = document.getElementById("status");
const prompt = document.getElementById("prompt");
const events = document.getElementById("events");
let overlayView = null;
let overlaySlug = null;
let overlayFrameId = -1;
const pressedKeys = new Map();
const pressedButtons = new Set();
const gamepadSnapshots = new Map();
let lastPointerPosition = {x: 0, y: 0};
let hideCursor = false;
let lockCursorToWindow = false;

const MAX_NONCRITICAL_BUFFER_BYTES = 4 * 1024;

let pendingPointerMove = null;
let pointerMoveHandle = null;
let pendingWheel = null;
let wheelHandle = null;

const cancelPendingPointerMove = () => {
  if (pointerMoveHandle !== null) {
    window.cancelAnimationFrame(pointerMoveHandle);
    pointerMoveHandle = null;
  }
  pendingPointerMove = null;
};

const showStatus = (message, isError = false) => {
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("error", isError);
};

const send = (payload, channel = controls) => {
  if (channel.readyState !== "open") {
    return false;
  }
  try {
    channel.send(JSON.stringify(payload));
    return true;
  } catch (error) {
    console.debug("Unable to send WebRTC control message.", error);
    return false;
  }
};

const sendInput = (
  payload,
  {
    channel = controls,
    dropIfCongested = false,
  } = {},
) => {
  if (dropIfCongested && channel.bufferedAmount > MAX_NONCRITICAL_BUFFER_BYTES) {
    return false;
  }
  return send(payload, channel);
};

// Tab close often never fires datachannel close, especially over TURN relay.
// The server pauses generation when this ping (or any other controls message)
// has been missing for a few intervals.
const HEARTBEAT_INTERVAL_MS = 1000;
const sendHeartbeat = () => {
  send({type: "ping"});
};
controls.addEventListener("open", () => {
  sendHeartbeat();
  window.setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);
});

const updateCursorVisibility = () => {
  video.style.cursor = hideCursor && document.hasFocus() ? "none" : "";
};

const applyCursorOptions = options => {
  hideCursor = options.hide_cursor === true;
  lockCursorToWindow = options.lock_cursor_to_window === true;
  updateCursorVisibility();
  if (!lockCursorToWindow && document.pointerLockElement === video) {
    document.exitPointerLock();
  }
};

window.addEventListener("focus", updateCursorVisibility);
window.addEventListener("blur", updateCursorVisibility);

controls.addEventListener("message", event => {
  if (typeof event.data !== "string") {
    return;
  }
  try {
    const payload = JSON.parse(event.data);
    if (payload?.type === "cursor_options") {
      applyCursorOptions(payload);
      return;
    }
    if (payload?.type === "session_metadata" && typeof payload.prompt === "string") {
      prompt.textContent = payload.prompt;
      overlayFrameId = -1;
      return;
    }
    if (payload?.type === "prompt" && Array.isArray(payload.events)) {
      // Each message carries one chunk's complete set. The next chunk's
      // message replaces it wholesale; nothing is blanked in between.
      if (typeof payload.static === "string" && payload.static) {
        prompt.textContent = payload.static;
      }
      events.replaceChildren(...payload.events.filter(line => typeof line === "string").map(line => {
        const item = document.createElement("div");
        item.textContent = line;
        return item;
      }));
      return;
    }
    if (payload?.type === "game_ui") {
      applyGameUi(payload);
      return;
    }
    if (payload?.type === "error") {
      console.warn(`WebRTC server: ${payload.message ?? "unknown error"}`);
    }
  } catch (error) {
    console.warn("Ignored malformed WebRTC control response.", error);
  }
});

peer.ontrack = event => {
  const target = event.transceiver === peer.getTransceivers()[0] ? video : document.getElementById("game-video");
  target.srcObject = new MediaStream([event.track]);
  target.play().catch(error => {
    showStatus(`Video playback failed: ${error.message}`, true);
  });
};

video.addEventListener("playing", () => {
  status.hidden = true;
});

peer.addEventListener("connectionstatechange", () => {
  if (peer.connectionState === "connected") {
    if (video.readyState < 2) {
      showStatus("Connected. Waiting for the first video frame…");
    } else {
      status.hidden = true;
    }
  } else if (["failed", "closed"].includes(peer.connectionState)) {
    showStatus(`WebRTC connection ${peer.connectionState}.`, true);
  }
});

window.addEventListener("keydown", event => {
  if (window.__PAIRED_VIDEO__) event.preventDefault();
  const keyId = event.code || event.key;
  if (pressedKeys.has(keyId)) {
    return;
  }
  const wasSent = sendInput({
    type: "keyboard",
    key: event.key,
    pressed: true,
  });
  if (wasSent) {
    pressedKeys.set(keyId, event.key);
  }
});

window.addEventListener("keyup", event => {
  const keyId = event.code || event.key;
  const pressedKey = pressedKeys.get(keyId);
  if (pressedKey === undefined) {
    return;
  }
  const wasSent = sendInput({
    type: "keyboard",
    key: pressedKey,
    pressed: false,
  });
  if (wasSent) {
    pressedKeys.delete(keyId);
  }
});

video.tabIndex = 0;

const renderedVideoBounds = () => {
  const bounds = video.getBoundingClientRect();
  if (!video.videoWidth || !video.videoHeight || !bounds.width || !bounds.height) {
    return bounds;
  }

  const scale = Math.min(
    bounds.width / video.videoWidth,
    bounds.height / video.videoHeight,
  );
  const width = video.videoWidth * scale;
  const height = video.videoHeight * scale;
  return {
    left: bounds.left + (bounds.width - width) / 2,
    top: bounds.top + (bounds.height - height) / 2,
    width,
    height,
  };
};

const layoutOverlay = () => {
  if (!overlayHost) {
    return;
  }
  const stage = video.parentElement?.getBoundingClientRect();
  const letterbox = renderedVideoBounds();
  if (!stage || !letterbox.width || !letterbox.height) {
    overlayHost.hidden = true;
    return;
  }
  overlayHost.hidden = overlayView === null;
  overlayHost.style.left = `${letterbox.left - stage.left}px`;
  overlayHost.style.top = `${letterbox.top - stage.top}px`;
  overlayHost.style.width = `${letterbox.width}px`;
  overlayHost.style.height = `${letterbox.height}px`;
};

const loadOverlay = async slug => {
  if (!overlayHost || !slug || slug === overlaySlug) {
    return overlayView;
  }
  const shadow = overlayHost.shadowRoot ?? overlayHost.attachShadow({mode: "open"});
  shadow.replaceChildren();
  overlayView = null;
  overlaySlug = slug;
  overlayFrameId = -1;
  const sheet = document.createElement("link");
  sheet.rel = "stylesheet";
  sheet.href = `/games/${encodeURIComponent(slug)}/hud/overlay.css`;
  const mountRoot = document.createElement("div");
  mountRoot.style.cssText = "position:absolute;inset:0;";
  shadow.append(sheet, mountRoot);
  const module = await import(`/games/${encodeURIComponent(slug)}/hud/overlay.js`);
  if (overlaySlug !== slug) {
    return null;
  }
  overlayView = module.mount(mountRoot);
  mountRoot.addEventListener("click", event => {
    const action = event.target.closest("[data-game-action]")?.dataset.gameAction;
    if (!action) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    sendInput({
      type: "client_action",
      channel: "game_ui",
      action,
    });
  });
  overlayHost.hidden = false;
  layoutOverlay();
  return overlayView;
};

const applyGameUi = payload => {
  // The workbench home screen watches this to know when a round ends and when
  // the first frame of a newly chosen game has arrived. The payload is the
  // server's own envelope, forwarded unchanged.
  window.dispatchEvent(new CustomEvent("game2v-ui", {detail: payload}));
  const slug = payload?.slug;
  const frameId = Number(payload?.frame_id);
  if (typeof slug !== "string" || !slug) {
    return;
  }
  if (Number.isFinite(frameId) && overlaySlug === slug && frameId === overlayFrameId) {
    return;
  }
  if (Number.isFinite(frameId)) {
    overlayFrameId = frameId;
  }
  loadOverlay(slug).then(view => {
    view?.update(payload.state);
    layoutOverlay();
  }).catch(error => {
    console.warn("Unable to load game overlay.", error);
  });
};

window.addEventListener("resize", layoutOverlay);
video.addEventListener("resize", layoutOverlay);
video.addEventListener("loadedmetadata", layoutOverlay);

const pointerPosition = event => {
  const bounds = renderedVideoBounds();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
    y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
  };
};

const activePointerPosition = event => {
  if (!lockCursorToWindow || document.pointerLockElement !== video) {
    return pointerPosition(event);
  }
  const bounds = renderedVideoBounds();
  return {
    x: lastPointerPosition.x + event.movementX / bounds.width,
    y: lastPointerPosition.y + event.movementY / bounds.height,
  };
};

const lockPointerToWindow = () => {
  if (!lockCursorToWindow || document.pointerLockElement === video) {
    return;
  }
  try {
    const request = video.requestPointerLock();
    request?.catch(error => {
      console.debug("Unable to lock the cursor to the window.", error);
    });
  } catch (error) {
    console.debug("Unable to lock the cursor to the window.", error);
  }
};

video.addEventListener("pointermove", event => {
  lastPointerPosition = activePointerPosition(event);
  pendingPointerMove = lastPointerPosition;
  if (pointerMoveHandle !== null) {
    return;
  }
  pointerMoveHandle = window.requestAnimationFrame(() => {
    pointerMoveHandle = null;
    const position = pendingPointerMove;
    pendingPointerMove = null;
    if (position === null) {
      return;
    }
    sendInput(
      {
        type: "mouse",
        action: "move",
        ...position,
      },
      {
        channel: pointerControls,
        dropIfCongested: true,
      },
    );
  });
});

video.addEventListener("pointerdown", event => {
  video.focus();
  if (lockCursorToWindow) {
    lockPointerToWindow();
  } else {
    video.setPointerCapture(event.pointerId);
  }
  cancelPendingPointerMove();
  lastPointerPosition = activePointerPosition(event);
  const wasSent = sendInput(
    {
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button: event.button,
      pressed: true,
    },
    {channel: pointerControls},
  );
  if (wasSent) {
    pressedButtons.add(event.button);
  }
  event.preventDefault();
});

video.addEventListener("pointerup", event => {
  if (!pressedButtons.has(event.button)) {
    return;
  }
  cancelPendingPointerMove();
  lastPointerPosition = activePointerPosition(event);
  const wasSent = sendInput(
    {
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button: event.button,
      pressed: false,
    },
    {channel: pointerControls},
  );
  if (wasSent) {
    pressedButtons.delete(event.button);
  }
  event.preventDefault();
});

video.addEventListener("pointercancel", () => {
  cancelPendingPointerMove();
  for (const button of [...pressedButtons]) {
    const wasSent = sendInput(
      {
        type: "mouse",
        action: "button",
        ...lastPointerPosition,
        button,
        pressed: false,
      },
      {channel: pointerControls},
    );
    if (wasSent) {
      pressedButtons.delete(button);
    }
  }
});

video.addEventListener("wheel", event => {
  const position = activePointerPosition(event);
  if (pendingWheel === null) {
    pendingWheel = {
      position,
      wheelX: 0,
      wheelY: 0,
    };
  }
  pendingWheel.position = position;
  pendingWheel.wheelX += -Math.sign(event.deltaX);
  pendingWheel.wheelY += -Math.sign(event.deltaY);
  if (wheelHandle === null) {
    wheelHandle = window.requestAnimationFrame(() => {
      wheelHandle = null;
      const wheel = pendingWheel;
      pendingWheel = null;
      if (wheel === null) {
        return;
      }
      sendInput(
        {
          type: "mouse",
          action: "wheel",
          ...wheel.position,
          wheel_x: wheel.wheelX,
          wheel_y: wheel.wheelY,
        },
        {
          channel: pointerControls,
          dropIfCongested: true,
        },
      );
    });
  }
  event.preventDefault();
}, {passive: false});

video.addEventListener("focus", () => {
  sendInput({type: "focus", focused: true});
});
video.addEventListener("blur", () => {
  sendInput({type: "focus", focused: false});
});

const touchPayload = (event, touch, action) => {
  const bounds = renderedVideoBounds();
  return {
    type: "touch",
    action,
    touch_id: touch.identifier,
    x: Math.min(1, Math.max(0, (touch.clientX - bounds.left) / bounds.width)),
    y: Math.min(1, Math.max(0, (touch.clientY - bounds.top) / bounds.height)),
    pressure: Math.min(1, Math.max(0, touch.force || 0)),
    primary: touch.identifier === event.touches[0]?.identifier,
  };
};

for (const [domEvent, action] of [
  ["touchstart", "start"],
  ["touchmove", "move"],
  ["touchend", "end"],
  ["touchcancel", "cancel"],
]) {
  video.addEventListener(domEvent, event => {
    for (const touch of event.changedTouches) {
      send(touchPayload(event, touch, action));
    }
    event.preventDefault();
  }, {passive: false});
}

const gamepadPayload = (gamepad, action = "state") => ({
  type: "gamepad",
  action,
  index: gamepad.index,
  id: gamepad.id,
  mapping: gamepad.mapping,
  axes: Array.from(gamepad.axes),
  buttons: gamepad.buttons.map(button => button.value),
  pressed: gamepad.buttons.map(button => button.pressed),
});

window.addEventListener("gamepadconnected", event => {
  gamepadSnapshots.delete(event.gamepad.index);
  send(gamepadPayload(event.gamepad, "connected"));
});

window.addEventListener("gamepaddisconnected", event => {
  gamepadSnapshots.delete(event.gamepad.index);
  send(gamepadPayload(event.gamepad, "disconnected"));
});

const pollGamepads = () => {
  for (const gamepad of navigator.getGamepads?.() || []) {
    if (!gamepad) {
      continue;
    }
    const payload = gamepadPayload(gamepad);
    const snapshot = JSON.stringify(payload);
    if (gamepadSnapshots.get(gamepad.index) !== snapshot) {
      gamepadSnapshots.set(gamepad.index, snapshot);
      send(payload);
    }
  }
  window.requestAnimationFrame(pollGamepads);
};
window.requestAnimationFrame(pollGamepads);

window.addEventListener("blur", () => {
  cancelPendingPointerMove();
  for (const [keyId, key] of [...pressedKeys]) {
    const wasSent = sendInput({
      type: "keyboard",
      key,
      pressed: false,
    });
    if (wasSent) {
      pressedKeys.delete(keyId);
    }
  }
  for (const button of [...pressedButtons]) {
    const wasSent = sendInput(
      {
        type: "mouse",
        action: "button",
        ...lastPointerPosition,
        button,
        pressed: false,
      },
      {channel: pointerControls},
    );
    if (wasSent) {
      pressedButtons.delete(button);
    }
  }
});

const waitForIceGatheringComplete = async () => {
  if (peer.iceGatheringState === "complete") {
    return;
  }
  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", onStateChange);
      reject(new Error("Timed out while gathering WebRTC network candidates."));
    }, 10000);
    const onStateChange = () => {
      if (peer.iceGatheringState === "complete") {
        window.clearTimeout(timeout);
        peer.removeEventListener("icegatheringstatechange", onStateChange);
        resolve();
      }
    };
    peer.addEventListener("icegatheringstatechange", onStateChange);
  });
};

const waitForServer = async () => {
  while (true) {
    try {
      const health = await fetch("/healthz", {cache: "no-store"});
      const state = health.ok ? await health.json() : null;
      if (state?.open) {
        return state;
      }
    } catch (error) {
      console.debug("WebRTC server is not ready yet.", error);
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
};

async function connect() {
  showStatus("Waiting for the server…");
  const server = await waitForServer();
  applyCursorOptions(server);
  showStatus("Gathering WebRTC network candidates…");
  await peer.setLocalDescription(await peer.createOffer());
  await waitForIceGatheringComplete();
  if (!peer.localDescription.sdp.includes("a=candidate:")) {
    throw new Error(iceServers.length
      ? "TURN relay unavailable: no WebRTC relay candidate was gathered. Check the TURN service and TCP port."
      : "No WebRTC network candidate was gathered. Check the browser's network access.");
  }
  const response = await fetch(
    "/api/webrtc/offer" + window.location.search,
    {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({...peer.localDescription.toJSON(), paired_video: window.__PAIRED_VIDEO__ === true}),
    },
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  showStatus("Connecting video…");
  await peer.setRemoteDescription(await response.json());
}

connect().catch(error => {
  console.error("Unable to start WebRTC.", error);
  showStatus(`Unable to start WebRTC: ${error.message}`, true);
});

(async () => {
  // The play page shows only the game; the workbench and panel extensions
  // would take the page over, so it opts out of them.
  if (window.__NO_EXTENSIONS__) return;
  try {
    const manifest = await (await fetch("/extensions.json", {cache: "no-store"})).json();
    for (const src of manifest.scripts ?? []) {
      try {
        await import(src);
      } catch (error) {
        console.warn(`Unable to load extension script ${src}.`, error);
      }
    }
  } catch (error) {
    console.warn("Unable to load /extensions.json.", error);
  }
})();
