(function () {
  "use strict";

  const page = document.body.dataset.page;
  const connectionElement = document.getElementById("connectionState");

  const phasePresentation = {
    STARTING: {
      title: "Preparing the robot",
      guidance: "Please wait while the system starts.",
      visual: "starting"
    },
    WAITING: {
      title: "Ready when you are",
      guidance: "Look toward the robot when you are ready.",
      visual: "waiting"
    },
    IDENTIFYING: {
      title: "Preparing your interaction",
      guidance: "Please remain naturally visible for a moment.",
      visual: "thinking"
    },
    GREETING: {
      title: "Welcome",
      guidance: "The robot is beginning the conversation.",
      visual: "speaking"
    },
    LISTENING: {
      title: "I’m listening",
      guidance: "Speak naturally and take your time.",
      visual: "listening"
    },
    TRANSCRIBING: {
      title: "Processing your speech",
      guidance: "Your speech is being converted to text.",
      visual: "thinking"
    },
    THINKING: {
      title: "Preparing an answer",
      guidance: "Please wait while the robot prepares its response.",
      visual: "thinking"
    },
    SPEAKING: {
      title: "Robot speaking",
      guidance: "The robot is responding to you.",
      visual: "speaking"
    },
    REOPENING: {
      title: "Would you like to continue?",
      guidance: "Say “one more question” to continue the conversation.",
      visual: "listening"
    },
    ENDING: {
      title: "Conversation complete",
      guidance: "Thank you for speaking with the robot.",
      visual: "waiting"
    },
    SHUTTING_DOWN: {
      title: "System stopping",
      guidance: "The robot is shutting down safely.",
      visual: "starting"
    },
    ERROR: {
      title: "The system needs attention",
      guidance: "Please wait for the researcher.",
      visual: "error"
    }
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value, fallback) {
    const element = byId(id);

    if (!element) {
      return;
    }

    const usable = value !== null && value !== undefined && value !== "";
    element.textContent = usable ? String(value) : fallback;
  }

  function titleCase(value) {
    return String(value || "")
      .toLowerCase()
      .replaceAll("_", " ")
      .replace(/\b\w/g, function (letter) {
        return letter.toUpperCase();
      });
  }

  function formatSeconds(value) {
    return typeof value === "number" ? value.toFixed(2) + " s" : "—";
  }

  function formatMilliseconds(value) {
    return typeof value === "number" ? value.toFixed(1) + " ms" : "—";
  }

  function formatBoolean(value) {
    if (value === true) {
      return "Available";
    }

    if (value === false) {
      return "Missing";
    }

    return "Unavailable";
  }

  function formatTimestamp(value) {
    if (!value) {
      return "Waiting for data";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
      return "Unavailable";
    }

    return date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });
  }

  function setConnection(connected) {
    if (!connectionElement) {
      return;
    }

    connectionElement.classList.toggle("connected", connected);
    connectionElement.classList.toggle("disconnected", !connected);
    connectionElement.lastChild.textContent = connected
      ? " Dashboard connected"
      : " Dashboard disconnected";
  }

  function renderVisitor(data) {
    const runtime = data.runtime;
    const conversation = data.conversation;
    const presentation =
      phasePresentation[runtime.phase] || phasePresentation.ERROR;

    const visual = byId("stateVisual");

    if (visual) {
      visual.dataset.state = presentation.visual;
    }

    setText("visitorPhase", presentation.title, "System state unavailable");
    setText("visitorGuidance", presentation.guidance, "Please wait.");
    setText(
      "visitorTranscript",
      conversation.transcript,
      "Your words will appear here."
    );
    setText(
      "visitorProfile",
      titleCase(conversation.speech_profile),
      "Normal"
    );
  }

  function moduleClass(value) {
    const status = String(value || "").toUpperCase();

    if (
      status.includes("READY") ||
      status.includes("ACTIVE") ||
      status.includes("PASS")
    ) {
      return "status-ready";
    }

    if (
      status.includes("ERROR") ||
      status.includes("FAILED")
    ) {
      return "status-error";
    }

    return "status-pending";
  }

  function renderModules(modules) {
    document.querySelectorAll("[data-module]").forEach(function (element) {
      const key = element.dataset.module;
      const value = modules[key];

      element.textContent = titleCase(value || "Unavailable");
      element.classList.remove(
        "status-ready",
        "status-pending",
        "status-error"
      );
      element.classList.add(moduleClass(value));
    });
  }

  function renderEvents(events) {
    const list = byId("eventList");

    if (!list) {
      return;
    }

    list.replaceChildren();

    if (!Array.isArray(events) || events.length === 0) {
      const empty = document.createElement("li");
      empty.className = "empty-event";
      empty.textContent = "Waiting for runtime events.";
      list.appendChild(empty);
      return;
    }

    events.forEach(function (event) {
      const item = document.createElement("li");
      const time = document.createElement("span");
      const category = document.createElement("span");
      const message = document.createElement("span");

      time.className = "event-time";
      category.className = "event-category";

      time.textContent = formatTimestamp(event.time);
      category.textContent = event.category || "system";
      message.textContent = event.message || "";

      item.append(time, category, message);
      list.appendChild(item);
    });
  }

  function renderResearch(data) {
    const runtime = data.runtime;
    const perception = data.perception;
    const identity = data.identity;
    const conversation = data.conversation;
    const metrics = data.metrics;

    setText("runtimePhase", titleCase(runtime.phase), "Unavailable");
    setText("runtimeMessage", runtime.message, "No runtime message");
    setText("identityStatus", titleCase(identity.status), "Not checked");
    setText(
      "recognitionSummary",
      typeof identity.recognition_ms === "number"
        ? identity.recognition_ms.toFixed(1) + " ms"
        : "No recognition measurement",
      "No recognition measurement"
    );
    setText(
      "researchProfile",
      titleCase(conversation.speech_profile),
      "Normal"
    );
    setText("completedTurns", conversation.completed_turns, "0");
    setText(
      "uptime",
      "Uptime " + Number(data.uptime_seconds || 0).toFixed(1) + " s",
      "Runtime starting"
    );

    setText("lastUpdated", formatTimestamp(data.updated_at), "Unavailable");
    setText("detectedPeople", perception.detected_people, "0");
    setText(
      "ownerTrack",
      perception.owner_track_id === null
        ? "None"
        : "Track " + perception.owner_track_id,
      "None"
    );
    setText("attentionStatus", titleCase(perception.attention), "Waiting");
    setText("livenessStatus", titleCase(perception.liveness), "Waiting");
    setText("engagementStatus", titleCase(perception.engagement), "Waiting");
    setText("faceStatus", formatBoolean(perception.face_available), "Unavailable");

    setText(
      "researchTranscript",
      conversation.transcript,
      "No accepted transcription yet."
    );
    setText(
      "researchResponse",
      conversation.response,
      "No response yet."
    );
    setText(
      "responseRoute",
      titleCase(conversation.route),
      "No route"
    );

    setText("recognitionMs", formatMilliseconds(identity.recognition_ms), "—");
    setText("whisperSeconds", formatSeconds(metrics.whisper_seconds), "—");
    setText(
      "speechEndTextSeconds",
      formatSeconds(metrics.speech_end_to_text_seconds),
      "—"
    );
    setText("openaiSeconds", formatSeconds(metrics.openai_seconds), "—");
    setText("piperSeconds", formatSeconds(metrics.piper_seconds), "—");
    setText(
      "responseStartSeconds",
      formatSeconds(metrics.response_start_seconds),
      "—"
    );
    setText(
      "playbackSeconds",
      formatSeconds(metrics.playback_seconds),
      "—"
    );

    renderModules(data.modules);
    renderEvents(data.events);
  }

  function render(data) {
    if (page === "visitor") {
      renderVisitor(data);
    } else if (page === "research") {
      renderResearch(data);
    }
  }

  async function refresh() {
    try {
      const response = await fetch("/api/state", {
        cache: "no-store",
        credentials: "same-origin"
      });

      if (!response.ok) {
        throw new Error("Dashboard state unavailable");
      }

      const data = await response.json();
      render(data);
      setConnection(true);
    } catch (_error) {
      setConnection(false);
    }
  }

  function startCameraRefresh() {
    const isResearch = page === "research";
    const camera = byId(isResearch ? "liveCamera" : "visitorCamera");
    const placeholder = byId(
      isResearch ? "cameraPlaceholder" : "visitorCameraPlaceholder"
    );
    const endpoint = isResearch
      ? "/api/frame.jpg"
      : "/api/visitor-frame.jpg";

    if (!camera) {
      return;
    }

    function requestNextFrame(delay) {
      window.setTimeout(function () {
        camera.src = endpoint + "?t=" + Date.now();
      }, delay);
    }

    camera.addEventListener("load", function () {
      camera.classList.add("ready");

      if (placeholder) {
        placeholder.classList.add("hidden");
      }

      requestNextFrame(160);
    });

    camera.addEventListener("error", function () {
      requestNextFrame(400);
    });

    requestNextFrame(0);
  }

  refresh();
  window.setInterval(refresh, 750);
  startCameraRefresh();
})();
