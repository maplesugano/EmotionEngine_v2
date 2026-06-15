import { useState, useMemo } from "react";
// @ts-expect-error no bundler types for plotly dist
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Plot = createPlotlyComponent(Plotly as any);

type EmotionProfile = Record<string, number>;
type MetaProjections = Record<string, number>; // "m1".."m5" → scalar

interface AnalyzeResponse {
  text: string;
  neutral_text: string;
  emotion_profile: EmotionProfile;
  meta_projections: MetaProjections;
}

// Projection of each meta-axis m_k onto each emotion direction r̂_e.
// Computed via: (meta_axes_pca @ r_hat.T), where r_hat = residualised CAA at layer 13.
// Recompute if CAA or meta-axis files change.
const META_AXIS_EMO_PROJ: Record<string, Record<string, number>> = {
  m1: { joy: 0.108596, trust: -0.124279, fear: -0.428254, surprise: 0.264737, sadness: 0.122753, disgust: -0.005559, anger: -0.004463, anticipation: 0.016462 },
  m2: { joy: 0.226891, trust: 0.296717,  fear: 0.094050,  surprise: -0.748511, sadness: -0.083725, disgust: -0.071294, anger: 0.009113, anticipation: 0.354620 },
  m3: { joy: 0.054389, trust: -0.191609, fear: -0.126749, surprise: 0.261490, sadness: 0.094247, disgust: 0.137215, anger: -0.039471, anticipation: -0.196183 },
  m4: { joy: -0.416222, trust: -0.353086, fear: 0.237689, surprise: -0.221549, sadness: 0.655307, disgust: 0.365089, anger: 0.299497, anticipation: -0.187663 },
  m5: { joy: -0.000241, trust: 0.273153, fear: 0.007457, surprise: -0.177427, sadness: -0.154406, disgust: -0.148823, anger: -0.049235, anticipation: 0.192601 },
};

const META_AXIS_LABELS: Record<string, { pos: string; neg: string }> = {
  m1: { pos: "agitated / reactive", neg: "mellow / accepting" },
  m2: { pos: "nostalgic / ruminative", neg: "pragmatic / forward" },
  m3: { pos: "interpersonally engaged", neg: "self-contained" },
};

const META_AXIS_COLORS: Record<string, string> = {
  m1: "#e53935",
  m2: "#43a047",
  m3: "#fb8c00",
};

const STABLE_META_AXES = ["m1", "m2", "m3"] as const;

// Empirical coherent windows from beta-sweep (soundness >= 0.70), Tab. exp15_n100_summary.
const COHERENT_WINDOW: Record<string, [number, number]> = {
  m1: [-8,  16],
  m2: [-16, 12],
  m3: [-12, 16],
};

const EMOTIONS = [
  "joy", "trust", "fear", "surprise",
  "sadness", "disgust", "anger", "anticipation",
];

const EMO_COLOR: Record<string, string> = {
  joy:          "#eaa406",
  fear:         "#663294",
  sadness:      "#23499c",
  disgust:      "#3e9a2d",
  anger:        "#9f0f06",
  anticipation: "#028a80",
  surprise:     "#cf2c89",
  trust:        "#d95a0d",
};

const NEUTRAL_BASELINE: Record<string, number> = {
  joy:          -0.838552,
  trust:        -0.870374,
  fear:         -0.835787,
  surprise:     -0.779176,
  sadness:      -0.824780,
  disgust:      -0.793761,
  anger:        -0.798691,
  anticipation: -0.853303,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jiggleScores(
  deviations: Record<string, number>,
  gammas: Record<string, number>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const e of EMOTIONS) {
    let s = deviations[e] ?? 0;
    for (const mk of STABLE_META_AXES) {
      s += (gammas[mk] ?? 0) * (META_AXIS_EMO_PROJ[mk][e] ?? 0);
    }
    out[e] = s;
  }
  return out;
}

function dominant(scores: Record<string, number>): string {
  return EMOTIONS.reduce((best, e) => scores[e] > scores[best] ? e : best, EMOTIONS[0]);
}

// ---------------------------------------------------------------------------
// Panel 1 — Emotion Map (2D: margin vs emotionality)
// ---------------------------------------------------------------------------
function EmotionMap({
  deviations, gammas,
}: { deviations: Record<string, number>; gammas: Record<string, number> }) {
  const baseScores = deviations;
  const shifted = jiggleScores(deviations, gammas);
  const anyGamma = STABLE_META_AXES.some(mk => (gammas[mk] ?? 0) !== 0);

  function mapPoint(scores: Record<string, number>) {
    const sorted = [...EMOTIONS].sort((a, b) => scores[b] - scores[a]);
    const dom = sorted[0], runner = sorted[1];
    const margin = scores[dom] - scores[runner];   // x: how stable
    const emotionality = scores[dom];              // y: how emotional
    return { x: margin, y: emotionality, dom };
  }

  const base = mapPoint(baseScores);
  const jig  = mapPoint(shifted);
  const flipped = anyGamma && jig.dom !== base.dom;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    // s(x) base
    {
      type: "scatter",
      mode: "markers+text",
      x: [base.x], y: [base.y],
      text: ["s(x)"],
      textposition: "top center",
      marker: { size: 14, color: EMO_COLOR[base.dom] ?? "#1976d2", symbol: "circle" },
      name: `s(x) [${base.dom}]`,
      hovertemplate: `margin: %{x:.3f}<br>emotionality: %{y:.3f}<br>dominant: ${base.dom}<extra></extra>`,
    },
    // jiggled point
    ...(anyGamma ? [{
      type: "scatter",
      mode: "markers+text",
      x: [jig.x], y: [jig.y],
      text: [flipped ? `⚠ ${jig.dom}` : "jiggled"],
      textposition: "bottom center",
      marker: { size: 10, color: flipped ? "#e53935" : EMO_COLOR[jig.dom] ?? "#555", symbol: "diamond" },
      name: flipped ? `jiggled [flipped→${jig.dom}]` : `jiggled [${jig.dom}]`,
      hovertemplate: `margin: %{x:.3f}<br>emotionality: %{y:.3f}<br>dominant: ${jig.dom}<extra></extra>`,
    }] : []),
    // flip boundary: vertical line at x=0
    {
      type: "scatter",
      mode: "lines",
      x: [0, 0], y: [-2, 2],
      line: { color: "#e53935", width: 1, dash: "dot" },
      name: "flip boundary",
      hoverinfo: "skip",
    },
  ];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const layout: any = {
    xaxis: {
      title: "margin (dominant − runner-up)",
      zeroline: true, zerolinecolor: "#e53935", zerolinewidth: 1,
      gridcolor: "#eee",
    },
    yaxis: { title: "emotionality (dominant score)", gridcolor: "#eee" },
    margin: { l: 52, r: 12, t: 12, b: 44 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "#fafafa",
    legend: { font: { size: 11 } },
    showlegend: true,
  };

  return (
    <Plot
      data={data}
      layout={layout}
      style={{ width: "100%", height: 280 }}
      config={{ responsive: true, displayModeBar: false }}
      useResizeHandler
    />
  );
}

// ---------------------------------------------------------------------------
// Panel 2 — Texture Bars (s(x)·m_k from meta_projections)
// ---------------------------------------------------------------------------
function TextureBars({ metaProjections, gammas }: {
  metaProjections: MetaProjections;
  gammas: Record<string, number>;
}) {
  return (
    <div>
      {STABLE_META_AXES.map(mk => {
        const proj = metaProjections[mk] ?? 0;
        const gamma = gammas[mk] ?? 0;
        const shifted = proj + gamma;  // approximate: s(x)·m_k + γ_k (same axis)
        const { pos, neg } = META_AXIS_LABELS[mk];
        const color = META_AXIS_COLORS[mk];

        // Normalise to [-1,1] display range (raw values are typically [-3, 3])
        const SCALE = 3;
        const clamp = (v: number) => Math.max(-1, Math.min(1, v / SCALE));
        const baseNorm    = clamp(proj);
        const shiftedNorm = clamp(shifted);

        return (
          <div key={mk} style={{ marginBottom: 18 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
              <span style={{ fontWeight: 700, color }}>{mk}</span>
              <span style={{ color: "#888", fontSize: 11 }}>
                {neg} ←→ {pos}
              </span>
              <span style={{ fontFamily: "monospace", fontSize: 11 }}>
                {proj.toFixed(2)}{gamma !== 0 ? ` → ${shifted.toFixed(2)}` : ""}
              </span>
            </div>
            {/* Two-sided bar centered at 0 */}
            <div style={{ position: "relative", height: 16, background: "#e8e8e8", borderRadius: 4 }}>
              {/* baseline marker */}
              <div style={{
                position: "absolute", top: 0, bottom: 0,
                left: `${(baseNorm + 1) / 2 * 100}%`,
                width: 3, background: color, opacity: 0.4, borderRadius: 2,
                transform: "translateX(-50%)",
              }} />
              {/* shifted marker */}
              <div style={{
                position: "absolute", top: 2, bottom: 2,
                left: `${(shiftedNorm + 1) / 2 * 100}%`,
                width: 5, background: color, borderRadius: 2,
                transform: "translateX(-50%)",
                transition: "left 0.15s ease",
              }} />
              {/* centre tick */}
              <div style={{
                position: "absolute", top: 0, bottom: 0,
                left: "50%", width: 1, background: "#bbb",
              }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#aaa", marginTop: 2 }}>
              <span>− (neg pole)</span>
              <span>+ (pos pole)</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel 3 — Competitor Margin Gauges
// ---------------------------------------------------------------------------
function MarginGauges({ deviations, gammas }: {
  deviations: Record<string, number>;
  gammas: Record<string, number>;
}) {
  const shifted = jiggleScores(deviations, gammas);
  const dom = dominant(shifted);
  const domScore = shifted[dom];
  const competitors = EMOTIONS.filter(e => e !== dom)
    .map(e => ({ e, margin: domScore - shifted[e] }))
    .sort((a, b) => a.margin - b.margin);

  const maxMargin = Math.max(...competitors.map(c => c.margin), 1);

  return (
    <div>
      <div style={{ fontSize: 12, color: "#666", marginBottom: 10 }}>
        Dominant: <strong style={{ color: EMO_COLOR[dom] }}>{dom}</strong>
        {" "}— distance to each competitor
      </div>
      {competitors.map(({ e, margin }) => {
        const pct = Math.min(100, (margin / maxMargin) * 100);
        const atRisk = margin < 0.05;
        const crossed = margin < 0;
        return (
          <div key={e} style={{ marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 2 }}>
              <span style={{ color: EMO_COLOR[e], fontWeight: 500, minWidth: 90 }}>{e}</span>
              <span style={{
                fontFamily: "monospace", fontSize: 11,
                color: crossed ? "#e53935" : atRisk ? "#f57c00" : "#555",
              }}>
                {crossed ? "⚠ " : ""}{margin.toFixed(3)}
              </span>
            </div>
            <div style={{ position: "relative", background: "#e8e8e8", borderRadius: 3, height: 10 }}>
              <div style={{
                position: "absolute", top: 0, bottom: 0, left: 0,
                width: `${Math.max(0, pct)}%`,
                background: crossed ? "#e53935" : atRisk ? "#f57c00" : EMO_COLOR[dom],
                borderRadius: 3,
                transition: "width 0.15s ease, background 0.15s ease",
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel 4 — γ-space 2D slice (γ1 vs γ2, γ3 fixed)
// ---------------------------------------------------------------------------
function GammaSlice({ deviations, gammas }: {
  deviations: Record<string, number>;
  gammas: Record<string, number>;
}) {
  const N = 40;  // grid resolution per axis

  // Fixed axis is m3; sweep over m1 (x) and m2 (y)
  const [L1, R1] = COHERENT_WINDOW["m1"];
  const [L2, R2] = COHERENT_WINDOW["m2"];
  const g3 = gammas["m3"] ?? 0;

  const baseDom = dominant(deviations);

  // Precompute the valid/flip grid
  const validX: number[] = [], validY: number[] = [];
  const flipX:  number[] = [], flipY:  number[] = [];

  for (let i = 0; i <= N; i++) {
    const g1 = L1 + (i / N) * (R1 - L1);
    for (let j = 0; j <= N; j++) {
      const g2 = L2 + (j / N) * (R2 - L2);
      const gs: Record<string, number> = { m1: g1, m2: g2, m3: g3 };
      const scores = jiggleScores(deviations, gs);
      if (dominant(scores) === baseDom) {
        validX.push(g1); validY.push(g2);
      } else {
        flipX.push(g1); flipY.push(g2);
      }
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    // valid region
    {
      type: "scatter",
      mode: "markers",
      x: validX, y: validY,
      marker: { size: 4, color: EMO_COLOR[baseDom] ?? "#1976d2", opacity: 0.25 },
      name: `${baseDom} (valid)`,
      hoverinfo: "skip",
    },
    // flip region
    ...(flipX.length > 0 ? [{
      type: "scatter",
      mode: "markers",
      x: flipX, y: flipY,
      marker: { size: 4, color: "#e53935", opacity: 0.15 },
      name: "flipped",
      hoverinfo: "skip",
    }] : []),
    // coherent window box outline
    {
      type: "scatter",
      mode: "lines",
      x: [L1, R1, R1, L1, L1],
      y: [L2, L2, R2, R2, L2],
      line: { color: "#999", width: 1, dash: "dash" },
      name: "coherent window",
      hoverinfo: "skip",
    },
    // current slider position
    {
      type: "scatter",
      mode: "markers",
      x: [gammas["m1"] ?? 0],
      y: [gammas["m2"] ?? 0],
      marker: { size: 10, color: "#222", symbol: "cross", line: { color: "#fff", width: 1 } },
      name: "current γ",
      hovertemplate: `γ₁=%{x:.1f}, γ₂=%{y:.1f}<extra></extra>`,
    },
  ];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const layout: any = {
    xaxis: { title: "γ₁ (m1: agitated ↔ mellow)", zeroline: true, zerolinecolor: "#ccc", gridcolor: "#eee" },
    yaxis: { title: "γ₂ (m2: nostalgic ↔ pragmatic)", zeroline: true, zerolinecolor: "#ccc", gridcolor: "#eee" },
    margin: { l: 52, r: 12, t: 12, b: 44 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "#fafafa",
    legend: { font: { size: 11 } },
    showlegend: true,
  };

  return (
    <div>
      <div style={{ fontSize: 11, color: "#999", marginBottom: 6 }}>
        γ₃ (m3) fixed at {g3.toFixed(1)} — coloured region = valid prototype cone
      </div>
      <Plot
        data={data}
        layout={layout}
        style={{ width: "100%", height: 300 }}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Gamma Sliders
// ---------------------------------------------------------------------------
function GammaSliders({ gammas, setGammas }: {
  gammas: Record<string, number>;
  setGammas: React.Dispatch<React.SetStateAction<Record<string, number>>>;
}) {
  return (
    <div>
      {STABLE_META_AXES.map(mk => {
        const [L, R] = COHERENT_WINDOW[mk];
        const g = gammas[mk] ?? 0;
        const color = META_AXIS_COLORS[mk];
        const { pos, neg } = META_AXIS_LABELS[mk];
        return (
          <div key={mk} style={{ marginBottom: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 3 }}>
              <span style={{ fontWeight: 700, color }}>{mk}</span>
              <span style={{ fontSize: 11, color: "#888" }}>{neg} ←→ {pos}</span>
              <span style={{ fontFamily: "monospace", fontSize: 12 }}>γ = {g.toFixed(1)}</span>
            </div>
            {/* coloured track: grey outside coherent window, coloured inside */}
            <div style={{ position: "relative" }}>
              <input
                type="range"
                min={-16} max={16} step={0.5}
                value={g}
                onChange={ev => setGammas(prev => ({ ...prev, [mk]: parseFloat(ev.target.value) }))}
                style={{ width: "100%", accentColor: color }}
              />
              {/* coherent window indicator */}
              <div style={{
                position: "absolute",
                top: 4,
                left:  `${((L + 16) / 32) * 100}%`,
                right: `${((16 - R) / 32) * 100}%`,
                height: 4,
                background: `${color}33`,
                borderRadius: 2,
                pointerEvents: "none",
              }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#bbb" }}>
              <span>−16</span>
              <span style={{ color: `${color}99`, fontSize: 10 }}>coherent [{L}, {R}]</span>
              <button
                onClick={() => setGammas(prev => ({ ...prev, [mk]: 0 }))}
                style={{ fontSize: 10, border: "none", background: "none", color: "#aaa", cursor: "pointer", padding: 0 }}
              >reset</button>
              <span>+16</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel 5 — Steer + generated text
// ---------------------------------------------------------------------------
async function fetchMedia(endpoint: string, passage: string) {
  const r = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: passage }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(String(d.detail ?? r.statusText));
  }
  return r.json();
}

function SteerPanel({ text, neutralText, gammas, wantPainting, wantMusic }: {
  text: string;
  neutralText: string;
  gammas: Record<string, number>;
  wantPainting: boolean;
  wantMusic: boolean;
}) {
  const [generated, setGenerated] = useState<string | null>(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [imageB64, setImageB64]   = useState<string | null>(null);
  const [audioB64, setAudioB64]   = useState<string | null>(null);
  const [mediaLoading, setMediaLoading] = useState(false);
  const [mediaError, setMediaError]     = useState<string | null>(null);

  const anyGamma = STABLE_META_AXES.some(mk => Math.abs(gammas[mk] ?? 0) > 0.01);

  async function generate() {
    setLoading(true);
    setError(null);
    setGenerated(null);
    setImageB64(null);
    setAudioB64(null);
    setMediaError(null);
    try {
      const res = await fetch("/steer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, neutral_text: neutralText, gammas }),
      });
      if (!res.ok) {
        const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
        throw new Error(String(detail));
      }
      const data = await res.json();
      setGenerated(data.generated_text);

      if ((wantPainting || wantMusic) && data.generated_text) {
        setMediaLoading(true);
        const tasks: Promise<void>[] = [];
        if (wantPainting) {
          tasks.push(
            fetchMedia("/generate-image", data.generated_text)
              .then((d: { image_b64: string }) => setImageB64(d.image_b64))
              .catch((err: unknown) => setMediaError(prev => [prev, `Image: ${err}`].filter(Boolean).join(" | ")))
          );
        }
        if (wantMusic) {
          tasks.push(
            fetchMedia("/generate-music", data.generated_text)
              .then((d: { audio_b64: string }) => setAudioB64(d.audio_b64))
              .catch((err: unknown) => setMediaError(prev => [prev, `Music: ${err}`].filter(Boolean).join(" | ")))
          );
        }
        await Promise.all(tasks);
        setMediaLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // Highlight words that appear in generated but not in original (rough diff)
  function renderDiff(original: string, steered: string) {
    const origWords = new Set(original.toLowerCase().match(/\w+/g) ?? []);
    const words = steered.split(/(\s+)/);
    return words.map((tok, i) => {
      const isNew = /\w/.test(tok) && !origWords.has(tok.toLowerCase());
      return (
        <span key={i} style={isNew ? { background: "#fff9c4", borderRadius: 2 } : undefined}>
          {tok}
        </span>
      );
    });
  }

  return (
    <div>
      <button
        onClick={generate}
        disabled={loading}
        style={{
          padding: "7px 20px", fontSize: 13,
          borderRadius: 6, border: "1px solid #1976d2",
          background: loading ? "#e3f2fd" : "#1976d2",
          color: loading ? "#1976d2" : "#fff",
          cursor: loading ? "wait" : "pointer",
          marginBottom: 14,
        }}
      >
        {loading ? "Generating…" : anyGamma ? "Generate steered text" : "Generate (γ = 0, no steering)"}
      </button>

      {error && <p style={{ color: "#c62828", fontSize: 13 }}>Error: {error}</p>}

      {generated !== null && generated === "" && (
        <p style={{ fontSize: 13, color: "#888", fontStyle: "italic" }}>
          (model produced no output — try different γ values or a longer input)
        </p>
      )}
      {generated !== null && generated !== "" && (
        <div>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 11, color: "#999", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>original</div>
            <div style={{ fontSize: 13, color: "#555", fontStyle: "italic", lineHeight: 1.6 }}>{text}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#999", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
              steered ({STABLE_META_AXES.filter(mk => Math.abs(gammas[mk] ?? 0) > 0.01).map(mk => `${mk}=${(gammas[mk] ?? 0).toFixed(1)}`).join(", ") || "no shift"})
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.6, color: "#222" }}>
              {renderDiff(text, generated)}
            </div>
          </div>
        </div>
      )}

      {(mediaLoading || imageB64 || audioB64 || mediaError) && (
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid #f0f0f0" }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#333", marginBottom: 8 }}>STEERED MEDIA</div>
          {mediaLoading && <p style={{ fontSize: 13, color: "#888" }}>Generating…</p>}
          {mediaError && <p style={{ color: "#c62828", fontSize: 13 }}>Error: {mediaError}</p>}
          {imageB64 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: "#999", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>painting</div>
              <img src={`data:image/png;base64,${imageB64}`} alt="AI oil painting" style={{ width: "100%", borderRadius: 6, display: "block" }} />
            </div>
          )}
          {audioB64 && (
            <div>
              <div style={{ fontSize: 11, color: "#999", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>music</div>
              <audio controls src={`data:audio/mpeg;base64,${audioB64}`} style={{ width: "100%" }} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------
export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gammas, setGammas] = useState<Record<string, number>>(
    Object.fromEntries(STABLE_META_AXES.map(mk => [mk, 0]))
  );
  const [wantPainting, setWantPainting] = useState(false);
  const [wantMusic, setWantMusic] = useState(false);
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [audioB64, setAudioB64] = useState<string | null>(null);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [mediaLoading, setMediaLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setImageB64(null);
    setAudioB64(null);
    setMediaError(null);
    try {
      const res = await fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
        throw new Error(String(detail));
      }
      const analysisResult: AnalyzeResponse = await res.json();
      setResult(analysisResult);
      setGammas(Object.fromEntries(STABLE_META_AXES.map(mk => [mk, 0])));

      if (wantPainting || wantMusic) {
        setMediaLoading(true);
        const passage = analysisResult.text;
        const tasks: Promise<void>[] = [];
        if (wantPainting) {
          tasks.push(
            fetchMedia("/generate-image", passage)
              .then((d: { image_b64: string }) => setImageB64(d.image_b64))
              .catch((err: unknown) => setMediaError(prev => [prev, `Image: ${err}`].filter(Boolean).join(" | ")))
          );
        }
        if (wantMusic) {
          tasks.push(
            fetchMedia("/generate-music", passage)
              .then((d: { audio_b64: string }) => setAudioB64(d.audio_b64))
              .catch((err: unknown) => setMediaError(prev => [prev, `Music: ${err}`].filter(Boolean).join(" | ")))
          );
        }
        await Promise.all(tasks);
        setMediaLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const deviationMap = useMemo(() => {
    const m: Record<string, number> = {};
    if (result) {
      for (const e of EMOTIONS) {
        m[e] = (result.emotion_profile[e] ?? 0) - (NEUTRAL_BASELINE[e] ?? 0);
      }
    }
    return m;
  }, [result]);

  const anyGamma = STABLE_META_AXES.some(mk => (gammas[mk] ?? 0) !== 0);
  const baseDom = result ? dominant(deviationMap) : null;
  const jigDom  = result ? dominant(jiggleScores(deviationMap, gammas)) : null;
  const flipped  = anyGamma && baseDom !== null && jigDom !== baseDom;

  const SECTION: React.CSSProperties = {
    background: "#fff",
    border: "1px solid #e8e8e8",
    borderRadius: 8,
    padding: "14px 16px",
    marginBottom: 16,
  };
  const H2: React.CSSProperties = { fontSize: 14, fontWeight: 700, marginBottom: 10, color: "#333" };
  const HINT: React.CSSProperties = { fontSize: 11, color: "#999", marginBottom: 10 };

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", padding: "0 16px", fontFamily: "sans-serif" }}>
      <h1 style={{ fontSize: 20, marginBottom: 2 }}>EmotionEngine — s(x)</h1>
      <p style={{ color: "#888", fontSize: 12, marginBottom: 20 }}>
        emotion profile · pre-verbal texture · prototype cone explorer
      </p>

      {/* Input */}
      <div style={SECTION}>
        <form onSubmit={handleSubmit}>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Enter text..."
            rows={3}
            style={{
              width: "100%", fontSize: 14, padding: 10,
              borderRadius: 6, border: "1px solid #ccc",
              resize: "vertical", boxSizing: "border-box",
            }}
          />
          <div style={{ display: "flex", alignItems: "center", gap: 20, marginTop: 10, flexWrap: "wrap" }}>
            <button
              type="submit"
              disabled={loading || !text.trim()}
              style={{
                padding: "7px 22px", fontSize: 13,
                cursor: loading ? "wait" : "pointer",
                borderRadius: 6, border: "none",
                background: "#1976d2", color: "#fff",
                opacity: !text.trim() ? 0.5 : 1,
              }}
            >
              {loading ? "Analysing…" : "Analyse"}
            </button>
            <label style={{ fontSize: 13, color: "#555", display: "flex", alignItems: "center", gap: 6, cursor: "default" }}>
              <input type="checkbox" checked disabled style={{ accentColor: "#1976d2" }} readOnly />
              text
            </label>
            <label style={{ fontSize: 13, color: "#555", display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={wantPainting}
                onChange={e => setWantPainting(e.target.checked)}
                style={{ accentColor: "#1976d2" }}
              />
              painting
            </label>
            <label style={{ fontSize: 13, color: "#555", display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={wantMusic}
                onChange={e => setWantMusic(e.target.checked)}
                style={{ accentColor: "#1976d2" }}
              />
              music
            </label>
          </div>
        </form>
        {error && <p style={{ color: "#c62828", marginTop: 10, fontSize: 13 }}>Error: {error}</p>}
        {result && (
          <p style={{ marginTop: 10, fontSize: 12, color: "#666" }}>
            <strong>neutral rewrite:</strong>{" "}
            <span style={{ fontStyle: "italic" }}>{result.neutral_text}</span>
          </p>
        )}
      </div>

      {/* Media output */}
      {(mediaLoading || imageB64 || audioB64 || mediaError) && (
        <div style={SECTION}>
          <div style={H2}>Generated media</div>
          {mediaLoading && <p style={{ fontSize: 13, color: "#888" }}>Generating…</p>}
          {mediaError && <p style={{ color: "#c62828", fontSize: 13 }}>Error: {mediaError}</p>}
          {imageB64 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: "#999", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>painting</div>
              <img
                src={`data:image/png;base64,${imageB64}`}
                alt="AI oil painting"
                style={{ width: "100%", borderRadius: 6, display: "block" }}
              />
            </div>
          )}
          {audioB64 && (
            <div>
              <div style={{ fontSize: 11, color: "#999", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>music</div>
              <audio
                controls
                src={`data:audio/mpeg;base64,${audioB64}`}
                style={{ width: "100%" }}
              />
            </div>
          )}
        </div>
      )}

      {result && (
        <>
          {/* Flip banner */}
          {flipped && (
            <div style={{
              marginBottom: 16, padding: "8px 14px", borderRadius: 6,
              background: "#ffebee", border: "1px solid #e53935",
              fontSize: 13, color: "#b71c1c",
            }}>
              ⚠ Dominant emotion flipped: <strong>{baseDom}</strong> → <strong>{jigDom}</strong> — outside prototype cone
            </div>
          )}

          {/* Row 1: Emotion Map | Texture Bars */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
            <div style={SECTION}>
              <div style={H2}>Emotion Map</div>
              <p style={HINT}>x = dominant margin (→ stable), y = emotionality. Left of 0 = flip.</p>
              <EmotionMap deviations={deviationMap} gammas={gammas} />
            </div>
            <div style={SECTION}>
              <div style={H2}>Pre-verbal Texture (s(x) · mₖ)</div>
              <p style={HINT}>
                Where s(x) sits on each meta-axis. Thin bar = base; thick bar = after γ shift.
              </p>
              <TextureBars metaProjections={result.meta_projections} gammas={gammas} />
            </div>
          </div>

          {/* Row 2: Margin Gauges | γ-space Slice */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
            <div style={SECTION}>
              <div style={H2}>Competitor Margins</div>
              <p style={HINT}>Distance from dominant to each competitor. Reaches 0 = flip.</p>
              <MarginGauges deviations={deviationMap} gammas={gammas} />
            </div>
            <div style={SECTION}>
              <div style={H2}>γ-space Cone Slice (γ₁ vs γ₂)</div>
              <p style={HINT}>
                Coloured = valid prototype region. Red = flipped. Dashed box = coherent window. Cross = current γ.
              </p>
              <GammaSlice deviations={deviationMap} gammas={gammas} />
            </div>
          </div>

          {/* Gamma sliders */}
          <div style={SECTION}>
            <div style={H2}>Meta-axis controls (γₖ)</div>
            <p style={HINT}>
              Shift s(x) along pre-verbal axes. Shaded region on each slider = coherent window [L, R].
            </p>
            <GammaSliders gammas={gammas} setGammas={setGammas} />
          </div>

          {/* Generated text */}
          <div style={SECTION}>
            <div style={H2}>Steered generation</div>
            <p style={HINT}>
              Injects Σ γₖ mₖ into layer 13 residual stream and generates. New words highlighted in yellow.
            </p>
            <SteerPanel
              text={result.text}
              neutralText={result.neutral_text}
              gammas={gammas}
              wantPainting={wantPainting}
              wantMusic={wantMusic}
            />
          </div>
        </>
      )}
    </div>
  );
}
