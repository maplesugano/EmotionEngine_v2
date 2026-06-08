import { useState } from "react";

type EmotionProfile = Record<string, number>;

interface AnalyzeResponse {
  text: string;
  neutral_text: string;
  emotion_profile: EmotionProfile;
}

const EMOTIONS = [
  "joy", "trust", "fear", "surprise",
  "sadness", "disgust", "anger", "anticipation",
];

const BAR_COLOR: Record<string, string> = {
  joy: "#f5c518",
  trust: "#4caf50",
  fear: "#9c27b0",
  surprise: "#ff9800",
  sadness: "#2196f3",
  disgust: "#795548",
  anger: "#f44336",
  anticipation: "#00bcd4",
};

function EmotionBar({ emotion, score, max }: { emotion: string; score: number; max: number }) {
  const pct = max > 0 ? (score / max) * 100 : 0;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 2 }}>
        <span style={{ textTransform: "capitalize", fontWeight: 500 }}>{emotion}</span>
        <span style={{ fontFamily: "monospace" }}>{score.toFixed(4)}</span>
      </div>
      <div style={{ background: "#e0e0e0", borderRadius: 4, height: 12 }}>
        <div
          style={{
            background: BAR_COLOR[emotion] ?? "#888",
            width: `${Math.max(pct, 0)}%`,
            height: "100%",
            borderRadius: 4,
            transition: "width 0.4s ease",
          }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

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
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const maxScore = result
    ? Math.max(...EMOTIONS.map((e) => result.emotion_profile[e] ?? 0))
    : 0;

  return (
    <div style={{ maxWidth: 600, margin: "60px auto", padding: "0 16px", fontFamily: "sans-serif" }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>EmotionEngine — s(x)</h1>
      <p style={{ color: "#666", fontSize: 13, marginBottom: 24 }}>
        入力テキストの感情プロファイルを取得します。
      </p>

      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="テキストを入力..."
          rows={4}
          style={{
            width: "100%",
            fontSize: 15,
            padding: 10,
            borderRadius: 6,
            border: "1px solid #ccc",
            resize: "vertical",
            boxSizing: "border-box",
          }}
        />
        <button
          type="submit"
          disabled={loading || !text.trim()}
          style={{
            marginTop: 10,
            padding: "8px 24px",
            fontSize: 14,
            cursor: loading ? "wait" : "pointer",
            borderRadius: 6,
            border: "none",
            background: "#1976d2",
            color: "#fff",
            opacity: !text.trim() ? 0.5 : 1,
          }}
        >
          {loading ? "解析中..." : "解析"}
        </button>
      </form>

      {error && (
        <p style={{ color: "#c62828", marginTop: 16, fontSize: 14 }}>エラー: {error}</p>
      )}

      {result && (
        <div style={{ marginTop: 28 }}>
          <div style={{ marginBottom: 16, fontSize: 13, color: "#555" }}>
            <strong>neutral rewrite:</strong>{" "}
            <span style={{ fontStyle: "italic" }}>{result.neutral_text}</span>
          </div>
          <h2 style={{ fontSize: 16, marginBottom: 14 }}>Emotion profile</h2>
          {EMOTIONS.map((e) => (
            <EmotionBar
              key={e}
              emotion={e}
              score={result.emotion_profile[e] ?? 0}
              max={maxScore}
            />
          ))}
        </div>
      )}
    </div>
  );
}
