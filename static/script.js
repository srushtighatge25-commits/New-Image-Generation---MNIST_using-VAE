(() => {
  let limit = 3;
  let z = [0, 0];
  let busy = false;
  let queued = false;
  let dragging = false;

  const $ = (id) => document.getElementById(id);
  const map = $("map");
  const mapImg = $("map-img");
  const marker = $("marker");
  const guideX = $("guide-x");
  const guideY = $("guide-y");
  const digit = $("digit");
  const z1 = $("z1");
  const z2 = $("z2");
  const z1Out = $("z1-out");
  const z2Out = $("z2-out");
  const statusEl = $("status");
  const grid = $("grid");

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const fmt = (v) => v.toFixed(2).replace("-", "\u2212");

  function setStatus(msg) {
    statusEl.textContent = msg || "";
  }

  function render() {
    const left = ((z[0] + limit) / (2 * limit)) * 100;
    const top = ((limit - z[1]) / (2 * limit)) * 100;
    marker.style.left = left + "%";
    marker.style.top = top + "%";
    guideX.style.left = left + "%";
    guideY.style.top = top + "%";
    z1.value = z[0];
    z2.value = z[1];
    z1Out.textContent = fmt(z[0]);
    z2Out.textContent = fmt(z[1]);
  }

  async function decode() {
    if (busy) {
      queued = true;
      return;
    }
    busy = true;
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ z }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      digit.src = data.image;
      setStatus("");
    } catch (err) {
      setStatus("Could not decode this point: " + err.message);
    } finally {
      busy = false;
      if (queued) {
        queued = false;
        decode();
      }
    }
  }

  function setZ(a, b) {
    z = [clamp(a, -limit, limit), clamp(b, -limit, limit)];
    render();
    decode();
  }

  function pointToZ(ev) {
    const r = map.getBoundingClientRect();
    const fx = clamp((ev.clientX - r.left) / r.width, 0, 1);
    const fy = clamp((ev.clientY - r.top) / r.height, 0, 1);
    return [-limit + 2 * limit * fx, limit - 2 * limit * fy];
  }

  map.addEventListener("pointerdown", (ev) => {
    dragging = true;
    map.setPointerCapture(ev.pointerId);
    map.focus({ preventScroll: true });
    setZ(...pointToZ(ev));
  });
  map.addEventListener("pointermove", (ev) => {
    if (dragging) setZ(...pointToZ(ev));
  });
  ["pointerup", "pointercancel"].forEach((name) =>
    map.addEventListener(name, () => {
      dragging = false;
    })
  );

  map.addEventListener("keydown", (ev) => {
    const step = ev.shiftKey ? 0.5 : 0.1;
    const moves = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    if (!moves[ev.key]) return;
    ev.preventDefault();
    setZ(z[0] + moves[ev.key][0], z[1] + moves[ev.key][1]);
  });

  z1.addEventListener("input", () => setZ(parseFloat(z1.value), z[1]));
  z2.addEventListener("input", () => setZ(z[0], parseFloat(z2.value)));

  $("random-point").addEventListener("click", () => {
    // Box-Muller: a point from N(0, I)
    const gauss = () =>
      Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    setZ(gauss(), gauss());
  });

  async function loadSamples() {
    const button = $("new-samples");
    button.disabled = true;
    try {
      const res = await fetch("/api/random?n=16");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      grid.replaceChildren(
        ...data.items.map((item) => {
          const tile = document.createElement("button");
          tile.type = "button";
          tile.className = "tile";
          tile.setAttribute("aria-label", `Inspect z = ${fmt(item.z[0])}, ${fmt(item.z[1])}`);

          const img = document.createElement("img");
          img.src = item.image;
          img.alt = "Generated digit";

          const caption = document.createElement("span");
          caption.textContent = `${fmt(item.z[0])}, ${fmt(item.z[1])}`;

          tile.append(img, caption);
          tile.addEventListener("click", () => {
            setZ(item.z[0], item.z[1]);
            $("map-title").scrollIntoView({ behavior: "smooth", block: "start" });
          });
          return tile;
        })
      );
    } catch (err) {
      setStatus("Could not load samples: " + err.message);
    } finally {
      button.disabled = false;
    }
  }

  $("new-samples").addEventListener("click", loadSamples);

  async function init() {
    try {
      const info = await (await fetch("/api/info")).json();
      limit = info.limit;
      $("foot").textContent =
        `Latent dimension ${info.latent_dim}. Decoder loaded from models/vae_decoder.keras.`;
      z1.min = z2.min = -limit;
      z1.max = z2.max = limit;

      const res = await fetch("/api/map");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      mapImg.src = data.image;
    } catch (err) {
      setStatus("Could not load the latent map: " + err.message);
    }
    render();
    decode();
    loadSamples();
  }

  init();
})();
