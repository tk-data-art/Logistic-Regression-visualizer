"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ReTip,
  ResponsiveContainer, AreaChart, Area, ReferenceLine,
} from "recharts";

// ── Math helpers ─────────────────────────────────────────────────────────────
const sigmoid = z => 1 / (1 + Math.exp(-z));
const bce = (y, yhat) => {
  const e = 1e-9;
  return -(y * Math.log(yhat + e) + (1 - y) * Math.log(1 - yhat + e));
};

const seededRng = (seed) => {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
};

const makeDataset = (seed = 42, spread = 2.5) => {
  const rng = seededRng(seed);
  return Array.from({ length: 80 }, (_, i) => {
    const cls = i < 40 ? 0 : 1;
    const cx = cls === 0 ? -1.5 : 1.5;
    return {
      x1: +(cx + (rng() - 0.5) * spread).toFixed(3),
      x2: +((rng() - 0.5) * spread).toFixed(3),
      y: cls,
    };
  });
};

const gdStep = (w1, w2, b, data, lr) => {
  let dw1 = 0, dw2 = 0, db = 0, loss = 0;
  const n = data.length;
  for (const { x1, x2, y } of data) {
    const yhat = sigmoid(w1 * x1 + w2 * x2 + b);
    const err = yhat - y;
    dw1 += err * x1; dw2 += err * x2; db += err;
    loss += bce(y, yhat);
  }
  return { w1: w1 - lr * dw1 / n, w2: w2 - lr * dw2 / n, b: b - lr * db / n, loss: loss / n };
};

const calcMetrics = (w1, w2, b, data, thresh = 0.5) => {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const { x1, x2, y } of data) {
    const pred = sigmoid(w1 * x1 + w2 * x2 + b) >= thresh ? 1 : 0;
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else fn++;
  }
  const acc = (tp + tn) / data.length;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
  return { tp, fp, tn, fn, acc, prec, rec, f1 };
};

const calcRoc = (w1, w2, b, data) => {
  const scores = data.map(({ x1, x2, y }) => ({ score: sigmoid(w1 * x1 + w2 * x2 + b), y }));
  return Array.from({ length: 101 }, (_, i) => {
    const t = i / 100;
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const { score, y } of scores) {
      const p = score >= t ? 1 : 0;
      if (p === 1 && y === 1) tp++; else if (p === 1 && y === 0) fp++;
      else if (p === 0 && y === 0) tn++; else fn++;
    }
    return {
      fpr: +(fp / (fp + tn + 1e-9)).toFixed(3),
      tpr: +(tp / (tp + fn + 1e-9)).toFixed(3),
    };
  }).sort((a, b) => a.fpr - b.fpr);
};

const calcPr = (w1, w2, b, data) => {
  const scores = data.map(({ x1, x2, y }) => ({ score: sigmoid(w1 * x1 + w2 * x2 + b), y }));
  return Array.from({ length: 101 }, (_, i) => {
    const t = i / 100;
    let tp = 0, fp = 0, fn = 0;
    for (const { score, y } of scores) {
      const p = score >= t ? 1 : 0;
      if (p === 1 && y === 1) tp++; else if (p === 1 && y === 0) fp++;
      else if (p === 0 && y === 1) fn++;
    }
    return {
      rec: +(tp / (tp + fn + 1e-9)).toFixed(3),
      prec: +(tp / (tp + fp + 1e-9)).toFixed(3),
    };
  }).sort((a, b) => a.rec - b.rec);
};

const calcAuc = (roc) => {
  let a = 0;
  for (let i = 1; i < roc.length; i++)
    a += Math.abs(roc[i].fpr - roc[i - 1].fpr) * (roc[i].tpr + roc[i - 1].tpr) / 2;
  return a;
};

// ── Datasets ──────────────────────────────────────────────────────────────────
const DATASETS = {
  exam: {
    label: "Exam Pass/Fail",
    f1: "Study Hours", f2: "Sleep Hours",
    cls0: "Fail", cls1: "Pass",
    useCase: "Education analytics",
    data: makeDataset(42),
    eli5: "Imagine you have student records. More study hours and good sleep = more likely to pass. The sigmoid turns those numbers into a probability of passing.",
    analogy: "The model is like a teacher who says: based on how hard you studied and how rested you are, I estimate a 78% chance you pass.",
  },
  spam: {
    label: "Spam Detection",
    f1: "Link Count", f2: "Urgency Score",
    cls0: "Legitimate", cls1: "Spam",
    useCase: "Email filtering",
    data: makeDataset(77),
    eli5: "Think of your inbox. Emails with lots of links and urgent language are probably spam. The model learns exactly where to draw that line.",
    analogy: "The model is like an experienced inbox manager who has seen thousands of emails and can spot spam patterns instantly.",
  },
  tumour: {
    label: "Tumour Diagnosis",
    f1: "Cell Size", f2: "Cell Shape",
    cls0: "Benign", cls1: "Malignant",
    useCase: "Medical diagnosis",
    data: makeDataset(13),
    eli5: "Doctors measure cell characteristics under a microscope. Abnormal size and irregular shape together are warning signs of malignancy.",
    analogy: "The model acts like a pathologist who has reviewed thousands of biopsies, translating measurements into a malignancy probability.",
  },
  loan: {
    label: "Loan Default",
    f1: "Income Score", f2: "Debt Ratio",
    cls0: "Repaid", cls1: "Default",
    useCase: "Credit risk",
    data: makeDataset(55),
    eli5: "Banks want to know who will repay. Low income and high existing debt = higher default risk. The model outputs a default probability for each applicant.",
    analogy: "Think of the model as a credit officer with access to thousands of past loan outcomes, calculating risk from financial signals.",
  },
};

// ── Colors ────────────────────────────────────────────────────────────────────
const C = {
  navy:    "#002147",
  navyMid: "#0a3060",
  red:     "#C41E3A",
  teal:    "#0072CE",
  gold:    "#B8860B",
  green:   "#2D7D46",
  white:   "#FFFFFF",
  offWhite:"#F7F6F3",
  gray100: "#F0EEE9",
  gray200: "#E0DDD6",
  gray300: "#C8C4BB",
  gray400: "#9A9590",
  gray500: "#6B6760",
  gray600: "#3D3A35",
};

// ── Global CSS ────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #F7F6F3; }
  input[type=range] { appearance: none; height: 3px; border-radius: 2px; outline: none; cursor: pointer; }
  input[type=range]::-webkit-slider-thumb { appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #002147; cursor: pointer; border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,33,71,0.3); }
  select { background: #fff; border: 1px solid #E0DDD6; padding: 6px 10px; font-family: "Source Sans 3", sans-serif; font-size: 12px; color: #002147; cursor: pointer; outline: none; border-radius: 3px; }
  .tip { position: relative; display: inline-block; }
  .tip .tipbox { display: none; position: absolute; bottom: 130%; left: 50%; transform: translateX(-50%); background: #002147; color: #fff; font-size: 11px; padding: 8px 12px; border-radius: 3px; z-index: 200; font-family: "Source Sans 3", sans-serif; line-height: 1.5; width: 200px; white-space: normal; pointer-events: none; }
  .tip:hover .tipbox { display: block; }
  button { font-family: "Source Sans 3", sans-serif; }
`;

// ── Shared Components ─────────────────────────────────────────────────────────
const Panel = ({ children, style = {} }) => (
  <div style={{ background: C.white, border: "1px solid " + C.gray200, boxShadow: "0 1px 6px rgba(0,33,71,0.06)", padding: "24px", marginBottom: 20, ...style }}>
    {children}
  </div>
);

const STitle = ({ children }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
    <div style={{ width: 3, height: 18, background: C.red }} />
    <h2 style={{ fontFamily: "Playfair Display, serif", fontSize: 16, fontWeight: 700, color: C.navy }}>{children}</h2>
  </div>
);

const KpiCard = ({ label, value, sub, accent = C.navy, tip = "" }) => (
  <div style={{ background: C.white, border: "1px solid " + C.gray200, borderTop: "3px solid " + accent, padding: "16px 18px", flex: 1, minWidth: 130, boxShadow: "0 1px 4px rgba(0,33,71,0.06)" }}>
    <div className="tip" style={{ fontSize: 11, fontWeight: 600, color: C.gray400, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, fontFamily: "Source Sans 3, sans-serif", borderBottom: tip ? "1px dashed " + C.gray300 : "none", display: "inline-block", cursor: tip ? "help" : "default" }}>
      {label}
      {tip && <span className="tipbox">{tip}</span>}
    </div>
    <div style={{ fontSize: 28, fontFamily: "Playfair Display, serif", fontWeight: 700, color: C.navy, lineHeight: 1 }}>{value}</div>
    {sub && <div style={{ fontSize: 11, color: C.gray400, marginTop: 5, fontFamily: "Source Sans 3, sans-serif" }}>{sub}</div>}
  </div>
);

const SliderRow = ({ label, value, min, max, step, onChange, fmt = v => v.toFixed(1), color = C.navy, tip = "" }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
      <span className="tip" style={{ fontSize: 11, fontWeight: 600, color: C.gray500, textTransform: "uppercase", letterSpacing: "0.08em", fontFamily: "Source Sans 3, sans-serif", borderBottom: tip ? "1px dashed " + C.gray300 : "none", cursor: tip ? "help" : "default" }}>
        {label}
        {tip && <span className="tipbox">{tip}</span>}
      </span>
      <span style={{ fontSize: 13, fontFamily: "Playfair Display, serif", fontWeight: 700, color }}>{fmt(value)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(+e.target.value)}
      style={{ width: "100%", background: "linear-gradient(to right," + color + " " + ((value - min) / (max - min) * 100) + "%," + C.gray200 + " 0%)" }} />
  </div>
);

const ttProps = {
  contentStyle: { background: C.navy, border: "none", borderRadius: 2, padding: "8px 12px", fontFamily: "Source Sans 3, sans-serif", fontSize: 11 },
  labelStyle: { color: C.gray300, fontSize: 10 },
  itemStyle: { color: C.white, fontWeight: 600 },
  cursor: { stroke: C.gray300, strokeWidth: 1, strokeDasharray: "4 4" },
};

const axP = { stroke: C.gray200, tick: { fill: C.gray400, fontSize: 10, fontFamily: "Source Sans 3, sans-serif" } };

// ── Heatmap Canvas ────────────────────────────────────────────────────────────
function HeatmapCanvas({ w1, w2, b, data, thresh, ds }) {
  const canvasRef = useRef(null);
  const [selected, setSelected] = useState(null);
  const W = 380, H = 300;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const x1v = (px / W) * 8 - 4;
        const x2v = 4 - (py / H) * 8;
        const p = sigmoid(w1 * x1v + w2 * x2v + b);
        const idx = (py * W + px) * 4;
        if (p < 0.5) {
          const t = p * 2;
          img.data[idx]   = Math.round(t * 255);
          img.data[idx+1] = Math.round(t * 255 + (1 - t) * 114);
          img.data[idx+2] = Math.round(255 * (1 - t * 0.19));
        } else {
          const t = (p - 0.5) * 2;
          img.data[idx]   = Math.round(t * 196 + (1 - t) * 255);
          img.data[idx+1] = Math.round(t * 30  + (1 - t) * 255);
          img.data[idx+2] = Math.round(t * 58  + (1 - t) * 255);
        }
        img.data[idx+3] = 170;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [w1, w2, b]);

  const toCx = (x1v) => ((x1v + 4) / 8) * W;
  const toCy = (x2v) => ((4 - x2v) / 8) * H;

  const bp = w2 !== 0 ? [
    { x: toCx(-4), y: toCy((-w1 * (-4) - b) / w2) },
    { x: toCx(4),  y: toCy((-w1 * 4  - b) / w2) },
  ] : null;

  return (
    <div style={{ position: "relative", display: "inline-block", border: "1px solid " + C.gray200 }}>
      <canvas ref={canvasRef} width={W} height={H} style={{ display: "block" }} />
      <svg width={W} height={H} style={{ position: "absolute", top: 0, left: 0, overflow: "hidden" }}>
        {[-3,-2,-1,0,1,2,3].map(v => (
          <g key={v}>
            <line x1={toCx(v)} y1={0} x2={toCx(v)} y2={H} stroke="rgba(0,0,0,0.08)" strokeWidth={0.5} />
            <line x1={0} y1={toCy(v)} x2={W} y2={toCy(v)} stroke="rgba(0,0,0,0.08)" strokeWidth={0.5} />
          </g>
        ))}
        <line x1={W/2} y1={0} x2={W/2} y2={H} stroke={C.gray300} strokeWidth={1} />
        <line x1={0} y1={H/2} x2={W} y2={H/2} stroke={C.gray300} strokeWidth={1} />
        {bp && <line x1={bp[0].x} y1={bp[0].y} x2={bp[1].x} y2={bp[1].y} stroke={C.navy} strokeWidth={2.5} strokeLinecap="round" />}
        {data.map((pt, i) => {
          const prob = sigmoid(w1 * pt.x1 + w2 * pt.x2 + b);
          const pred = prob >= thresh ? 1 : 0;
          const ok = pred === pt.y;
          const cx = toCx(pt.x1), cy = toCy(pt.x2);
          return (
            <circle key={i} cx={cx} cy={cy} r={5.5}
              fill={pt.y === 1 ? C.teal : C.red}
              fillOpacity={ok ? 0.9 : 0.25}
              stroke={ok ? C.white : "#FF6600"}
              strokeWidth={ok ? 1 : 2}
              style={{ cursor: "pointer" }}
              onClick={() => setSelected(selected && selected.i === i ? null : { ...pt, i, pred, prob })}
            />
          );
        })}
      </svg>
      <div style={{ position: "absolute", bottom: 6, left: 6, display: "flex", gap: 8, background: "rgba(255,255,255,0.92)", padding: "4px 8px", border: "1px solid " + C.gray200, fontSize: 10, fontFamily: "Source Sans 3, sans-serif" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: C.teal, display: "inline-block" }} />{ds.cls1}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: C.red, display: "inline-block" }} />{ds.cls0}</span>
        <span style={{ color: C.gray400 }}>Faded = wrong</span>
      </div>
      <div style={{ position: "absolute", bottom: -18, left: "50%", transform: "translateX(-50%)", fontSize: 10, color: C.gray400, fontFamily: "Source Sans 3, sans-serif", whiteSpace: "nowrap" }}>{ds.f1}</div>
      {selected && (
        <div style={{ position: "absolute", top: 6, right: 6, background: C.navy, color: C.white, padding: "10px 12px", borderRadius: 3, fontSize: 11, fontFamily: "Source Sans 3, sans-serif", lineHeight: 1.8, minWidth: 168, zIndex: 10 }}>
          <div style={{ fontWeight: 700, borderBottom: "1px solid rgba(255,255,255,0.15)", paddingBottom: 4, marginBottom: 6 }}>Point Inspector</div>
          <div>{ds.f1}: <strong>{selected.x1}</strong></div>
          <div>{ds.f2}: <strong>{selected.x2}</strong></div>
          <div>True label: <strong>{selected.y === 1 ? ds.cls1 : ds.cls0}</strong></div>
          <div>Predicted: <strong style={{ color: selected.pred === selected.y ? "#7dffb0" : "#ffaa55" }}>{selected.pred === 1 ? ds.cls1 : ds.cls0}</strong></div>
          <div>P(positive): <strong>{selected.prob.toFixed(3)}</strong></div>
          <div>Loss: <strong>{bce(selected.y, selected.prob).toFixed(4)}</strong></div>
          <button onClick={() => setSelected(null)} style={{ marginTop: 6, background: "transparent", border: "1px solid rgba(255,255,255,0.3)", color: C.white, fontSize: 10, padding: "2px 8px", cursor: "pointer", borderRadius: 2 }}>Close</button>
        </div>
      )}
    </div>
  );
}

// ── TAB I: UNDERSTAND ─────────────────────────────────────────────────────────
function UnderstandTab({ mode, ds }) {
  const [w, setW] = useState(1);
  const [b, setB] = useState(0);

  const sigData = Array.from({ length: 100 }, (_, i) => {
    const x = -5 + i * 0.1;
    return { x: +x.toFixed(1), y: +sigmoid(w * x + b).toFixed(4) };
  });
  const decX = w !== 0 ? (-b / w).toFixed(2) : "n/a";
  const loss0 = Array.from({ length: 98 }, (_, i) => ({ yhat: +(0.01 + i * 0.01).toFixed(2), loss: +bce(0, 0.01 + i * 0.01).toFixed(4) }));
  const loss1 = Array.from({ length: 98 }, (_, i) => ({ yhat: +(0.01 + i * 0.01).toFixed(2), loss: +bce(1, 0.01 + i * 0.01).toFixed(4) }));

  return (
    <div>
      {/* Story card */}
      <Panel>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div style={{ flex: 2, minWidth: 280 }}>
            <STitle>What is Logistic Regression?</STitle>
            {mode === "beginner" ? (
              <>
                <div style={{ background: C.offWhite, borderLeft: "4px solid " + C.teal, padding: "12px 16px", marginBottom: 14, fontFamily: "Source Sans 3, sans-serif", fontSize: 13, lineHeight: 1.8, color: C.gray600 }}>
                  <strong style={{ color: C.navy }}>Think of it like a voting machine.</strong> The more evidence stacks up for one outcome, the more confidently the model votes for it. At the midpoint, it is a 50/50 coin flip.
                </div>
                <div style={{ fontFamily: "Source Sans 3, sans-serif", fontSize: 13, color: C.gray600, lineHeight: 1.8 }}>
                  <strong style={{ color: C.navy }}>Your dataset: </strong>{ds.label} -- {ds.useCase}
                </div>
                <div style={{ marginTop: 10, fontFamily: "Source Sans 3, sans-serif", fontSize: 13, color: C.gray500, lineHeight: 1.8, fontStyle: "italic" }}>
                  {ds.eli5}
                </div>
              </>
            ) : (
              <>
                <div style={{ background: C.offWhite, borderLeft: "4px solid " + C.teal, padding: "12px 16px", marginBottom: 14, fontFamily: "Source Sans 3, sans-serif", fontSize: 13, lineHeight: 1.8, color: C.gray600 }}>
                  Logistic regression models the log-odds of a binary outcome as a linear combination of features: <strong style={{ color: C.navy }}>log(p/(1-p)) = w1*x1 + w2*x2 + b</strong>. The sigmoid maps this logit into a probability in (0, 1).
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 12, background: C.offWhite, padding: "10px 14px", color: C.gray600, lineHeight: 1.9 }}>
                  <div>P(y=1|x) = sigma(w^T * x + b)</div>
                  <div>Loss = -(1/n) * sum[ y*log(p) + (1-y)*log(1-p) ]</div>
                  <div>Update: w := w - alpha * (1/n) * sum[ (p_i - y_i) * x_i ]</div>
                </div>
              </>
            )}
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ background: C.offWhite, padding: "14px 16px", border: "1px solid " + C.gray200 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: C.gray400, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 12, fontFamily: "Source Sans 3, sans-serif" }}>Learning Roadmap</div>
              {[
                { n: "1", label: "Understand", sub: "Sigmoid & Loss", color: C.teal, active: true },
                { n: "2", label: "Train", sub: "Gradient Descent", color: C.navy },
                { n: "3", label: "Evaluate", sub: "Metrics & ROC", color: C.red },
              ].map(item => (
                <div key={item.n} style={{ display: "flex", gap: 10, marginBottom: 12, alignItems: "flex-start" }}>
                  <div style={{ width: 24, height: 24, borderRadius: "50%", background: item.active ? item.color : C.gray200, color: item.active ? C.white : C.gray400, fontSize: 11, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontFamily: "Source Sans 3, sans-serif" }}>{item.n}</div>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: item.active ? item.color : C.gray500, fontFamily: "Source Sans 3, sans-serif" }}>{item.label}</div>
                    <div style={{ fontSize: 11, color: C.gray400, fontFamily: "Source Sans 3, sans-serif" }}>{item.sub}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Panel>

      {/* Sigmoid explorer + BCE loss */}
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <Panel style={{ flex: 2, minWidth: 320 }}>
          <STitle>Sigmoid Function Explorer</STitle>
          {mode === "beginner" ? (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, marginBottom: 14, fontFamily: "Source Sans 3, sans-serif" }}>
              The S-curve maps any input to a probability 0-100%. Drag <strong style={{ color: C.navy }}>weight</strong> to change the steepness and <strong style={{ color: C.teal }}>bias</strong> to shift it left or right. The red dashed line is the decision boundary -- where the model is exactly 50/50.
            </p>
          ) : (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, marginBottom: 14, fontFamily: "Source Sans 3, sans-serif" }}>
              sigma(z) = 1/(1+e^(-z)) where z = w*x + b. The inflection is at x = -b/w. d(sigma)/dz = sigma*(1-sigma) -- maximum slope 0.25 at z=0. Weight controls steepness; bias translates the boundary.
            </p>
          )}
          <ResponsiveContainer width="100%" height={210}>
            <LineChart data={sigData} margin={{ top: 8, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={C.gray100} />
              <XAxis dataKey="x" {...axP} tickCount={7} label={{ value: ds.f1 + " (scaled)", fill: C.gray400, fontSize: 10, position: "insideRight", offset: 16 }} />
              <YAxis {...axP} domain={[0, 1]} tickCount={5} label={{ value: "P(" + ds.cls1 + ")", fill: C.gray400, fontSize: 10, angle: -90, position: "insideLeft", offset: 10 }} />
              <ReferenceLine y={0.5} stroke={C.gray300} strokeDasharray="4 4" />
              {w !== 0 && <ReferenceLine x={+decX} stroke={C.red} strokeWidth={1.5} strokeDasharray="4 4" label={{ value: "boundary=" + decX, fill: C.red, fontSize: 9, position: "top", fontFamily: "Source Sans 3, sans-serif" }} />}
              <ReTip {...ttProps} formatter={v => [v.toFixed(4), "Probability"]} />
              <Line type="monotone" dataKey="y" stroke={C.navy} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: 20, marginTop: 12 }}>
            <div style={{ flex: 1 }}>
              <SliderRow label="Weight (w)" value={w} min={-4} max={4} step={0.1} onChange={setW} color={C.navy} tip={mode === "beginner" ? "Controls steepness of the curve. Larger = model is more decisive." : "Coefficient scaling the feature in the logit z = w*x + b. Controls d(sigma)/dx at the boundary."} />
              <SliderRow label="Bias (b)" value={b} min={-4} max={4} step={0.1} onChange={setB} color={C.teal} tip={mode === "beginner" ? "Shifts the curve left or right -- moves where the 50% crossover happens." : "Intercept term. Decision boundary location x0 = -b/w. Shifts the logit without changing slope."} />
            </div>
            <div style={{ flex: 1, background: C.offWhite, padding: "12px 14px", borderLeft: "3px solid " + C.red, fontSize: 12, fontFamily: "Source Sans 3, sans-serif", lineHeight: 2, color: C.gray600 }}>
              {mode === "beginner" ? (
                <>
                  <div>50% crossover at <strong style={{ color: C.red }}>x = {decX}</strong></div>
                  <div>At x=0: P = <strong style={{ color: C.navy }}>{sigmoid(b).toFixed(3)}</strong></div>
                  <div style={{ marginTop: 6, fontSize: 11, color: C.gray400 }}>
                    {w > 0.1 ? "Higher " + ds.f1 + " = more likely " + ds.cls1 : w < -0.1 ? "Lower " + ds.f1 + " = more likely " + ds.cls1 : "Feature has near-zero influence"}
                  </div>
                </>
              ) : (
                <>
                  <div>z = {w.toFixed(1)}*x + {b.toFixed(1)}</div>
                  <div>sigma(b) = {sigmoid(b).toFixed(4)}</div>
                  <div>x0 = {decX}</div>
                  <div>max slope = {(sigmoid(b) * (1 - sigmoid(b))).toFixed(4)}</div>
                </>
              )}
            </div>
          </div>
        </Panel>

        <Panel style={{ flex: 1, minWidth: 280 }}>
          <STitle>Log-Loss (BCE)</STitle>
          {mode === "beginner" ? (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, marginBottom: 12, fontFamily: "Source Sans 3, sans-serif" }}>
              Being <strong style={{ color: C.red }}>confidently wrong</strong> is punished severely. The loss curve shoots to infinity when the model is 100% wrong. Being correctly uncertain costs almost nothing.
            </p>
          ) : (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, marginBottom: 12, fontFamily: "Source Sans 3, sans-serif" }}>
              L = -[y*log(p) + (1-y)*log(1-p)]. As p approaches 0 for y=1, L approaches infinity. Gradient dL/dw = (p-y)*x makes parameters move toward reducing confident errors.
            </p>
          )}
          {[{ label: "True label: " + ds.cls1, d: loss1, color: C.teal }, { label: "True label: " + ds.cls0, d: loss0, color: C.red }].map(({ label, d, color }) => (
            <div key={label} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6, fontFamily: "Source Sans 3, sans-serif" }}>{label}</div>
              <ResponsiveContainer width="100%" height={110}>
                <AreaChart data={d} margin={{ top: 2, right: 8, bottom: 16, left: 0 }}>
                  <defs>
                    <linearGradient id={"lg" + label.replace(/\s/g, "")} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={color} stopOpacity={0.15} />
                      <stop offset="95%" stopColor={color} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="2 4" stroke={C.gray100} />
                  <XAxis dataKey="yhat" {...axP} tickCount={5} label={{ value: "predicted prob", fill: C.gray400, fontSize: 9, position: "insideBottomRight", offset: -4 }} />
                  <YAxis {...axP} tickCount={3} />
                  <ReTip {...ttProps} formatter={v => [v.toFixed(4), "Loss"]} />
                  <Area type="monotone" dataKey="loss" stroke={color} fill={"url(#lg" + label.replace(/\s/g, "") + ")"} strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ))}
        </Panel>
      </div>

      {/* Analogy cards -- beginner only */}
      {mode === "beginner" && (
        <Panel>
          <STitle>Key Analogies</STitle>
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            {[
              { title: "Sigmoid = Voting Machine", body: "Evidence accumulates on one side until the model votes with near-certainty. At zero evidence, it is a coin flip.", color: C.teal },
              { title: "Weight = Feature Importance", body: "A large weight means that feature heavily influences the prediction. Near zero = the feature barely matters.", color: C.navy },
              { title: "Bias = Default Tendency", body: "If you knew absolutely nothing about a case, the bias tells you which class to lean toward by default.", color: C.gold },
              { title: "Loss = Penalty Score", body: "The model is harshly penalized for being confidently wrong. Small, uncertain mistakes cost almost nothing.", color: C.red },
              { title: "Gradient = Compass", body: "The gradient tells the model which direction to adjust its weights to reduce the penalty score on the next step.", color: C.green },
            ].map(card => (
              <div key={card.title} style={{ flex: 1, minWidth: 180, borderTop: "3px solid " + card.color, background: C.offWhite, padding: "12px 14px" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: C.navy, fontFamily: "Playfair Display, serif", marginBottom: 5 }}>{card.title}</div>
                <div style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, fontFamily: "Source Sans 3, sans-serif" }}>{card.body}</div>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

// ── TAB II: TRAIN ─────────────────────────────────────────────────────────────
function TrainTab({ mode, ds, onTrained }) {
  const wRef = useRef({ w1: -2.5, w2: 2.5, b: -1.0 });
  const stepRef = useRef(0);
  const cancelRef = useRef(false);
  const lrRef = useRef(0.5);

  const [disp, setDisp] = useState({ w1: -2.5, w2: 2.5, b: -1.0 });
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState([]);
  const [step, setStep] = useState(0);
  const [log, setLog] = useState(["Dataset loaded: " + ds.label + ". Press Run to start training."]);
  const [thresh, setThresh] = useState(0.5);
  const [lr, setLr] = useState(0.5);
  const [verdict, setVerdict] = useState(null);

  const MAX = 200;

  const doReset = useCallback(() => {
    cancelRef.current = true;
    setRunning(false);
    wRef.current = { w1: -2.5, w2: 2.5, b: -1.0 };
    stepRef.current = 0;
    setDisp({ w1: -2.5, w2: 2.5, b: -1.0 });
    setHistory([]);
    setStep(0);
    setVerdict(null);
    setLog(["Model reset. Press Run to start training."]);
  }, []);

  useEffect(() => {
    doReset();
    setLog(["Dataset switched to: " + ds.label + ". Press Run to train."]);
  }, [ds]);

  useEffect(() => {
    lrRef.current = lr;
  }, [lr]);

  const startTraining = () => {
    if (stepRef.current >= MAX) return;
    cancelRef.current = false;
    setRunning(true);

    const tick = () => {
      if (cancelRef.current) return;
      if (stepRef.current >= MAX) {
        setRunning(false);
        const { w1, w2, b } = wRef.current;
        const m = calcMetrics(w1, w2, b, ds.data, 0.5);
        setVerdict({ w1, w2, b, ...m });
        onTrained({ w1, w2, b });
        setLog(prev => [...prev.slice(-9), "Training complete! Acc: " + (m.acc * 100).toFixed(1) + "% | F1: " + (m.f1 * 100).toFixed(1) + "%"]);
        return;
      }
      const { w1, w2, b } = wRef.current;
      const next = gdStep(w1, w2, b, ds.data, lrRef.current);
      wRef.current = { w1: next.w1, w2: next.w2, b: next.b };
      stepRef.current += 1;
      setDisp({ w1: next.w1, w2: next.w2, b: next.b });
      setStep(stepRef.current);
      setHistory(h => [...h, { step: stepRef.current, loss: +next.loss.toFixed(5) }]);
      if (stepRef.current % 25 === 0) {
        setLog(prev => [...prev.slice(-9), "Step " + stepRef.current + " / " + MAX + " -- loss: " + next.loss.toFixed(4)]);
      }
      setTimeout(tick, 45);
    };
    setTimeout(tick, 45);
  };

  const pauseTraining = () => {
    cancelRef.current = true;
    setRunning(false);
  };

  const m = calcMetrics(disp.w1, disp.w2, disp.b, ds.data, thresh);
  const currentLoss = history.at(-1)?.loss;
  const improvement = history.length > 2
    ? ((history[0].loss - history.at(-1).loss) / history[0].loss * 100).toFixed(1)
    : null;

  const lrWarning = lr >= 2.5;
  const lrSlow = lr <= 0.08;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
        <KpiCard label="Iteration" value={step + " / " + MAX} sub="Gradient descent steps" accent={C.navy} />
        <KpiCard label="Current Loss" value={currentLoss ? currentLoss.toFixed(4) : "--"} sub="Binary cross-entropy" accent={C.red} tip={mode === "beginner" ? "How wrong the model currently is. Lower = better. Watch it drop as training runs!" : "Mean BCE: -(1/n)*sum[y*log(p)+(1-y)*log(1-p)]"} />
        <KpiCard label="Accuracy" value={(m.acc * 100).toFixed(1) + "%"} sub={Math.round(m.acc * ds.data.length) + " / " + ds.data.length + " correct"} accent={C.teal} />
        {improvement && <KpiCard label="Loss Reduced" value={improvement + "%"} sub="From initial" accent={C.gold} />}
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {/* Heatmap panel */}
        <Panel style={{ flex: 2, minWidth: 320 }}>
          <STitle>Decision Boundary with Probability Heatmap</STitle>
          {mode === "beginner" ? (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, marginBottom: 14, fontFamily: "Source Sans 3, sans-serif" }}>
              <strong style={{ color: C.teal }}>Blue</strong> = confident this is {ds.cls1}. <strong style={{ color: C.red }}>Red</strong> = confident this is {ds.cls0}. White = 50/50 unsure. The dark line is the decision boundary. <strong>Click any dot</strong> to inspect it!
            </p>
          ) : (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, marginBottom: 14, fontFamily: "Source Sans 3, sans-serif" }}>
              Heatmap shows P(y=1|x1,x2) = sigma(w1*x1 + w2*x2 + b) across all feature space. Boundary is where P = threshold. Faded points with orange outline are misclassifications. Click any point to inspect its logit, probability, and individual loss.
            </p>
          )}
          <div style={{ overflowX: "auto", paddingBottom: 22 }}>
            <HeatmapCanvas w1={disp.w1} w2={disp.w2} b={disp.b} data={ds.data} thresh={thresh} ds={ds} />
          </div>
          <div style={{ marginTop: 16 }}>
            <SliderRow label="Classification Threshold" value={thresh} min={0.05} max={0.95} step={0.05} onChange={setThresh} color={C.gold} fmt={v => v.toFixed(2)} tip={mode === "beginner" ? "How confident must the model be before it predicts positive? 0.5 = balanced. Lower = catch more positives. Higher = fewer false alarms." : "tau: classify as positive if P(y=1|x) >= tau. Controls precision vs recall tradeoff."} />
          </div>
        </Panel>

        {/* Controls */}
        <div style={{ flex: 1, minWidth: 260 }}>
          <Panel>
            <STitle>Training Controls</STitle>
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              <button
                onClick={running ? pauseTraining : startTraining}
                disabled={step >= MAX}
                style={{ flex: 1, padding: "10px 0", background: running ? C.offWhite : C.navy, border: "1px solid " + C.navy, color: running ? C.navy : C.white, fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", cursor: step >= MAX ? "not-allowed" : "pointer", opacity: step >= MAX ? 0.5 : 1 }}>
                {running ? "Pause" : step === 0 ? "Run" : step >= MAX ? "Done" : "Resume"}
              </button>
              <button onClick={doReset} style={{ flex: 1, padding: "10px 0", background: C.offWhite, border: "1px solid " + C.gray300, color: C.gray600, fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", cursor: "pointer" }}>Reset</button>
            </div>

            <SliderRow label="Learning Rate" value={lr} min={0.01} max={3.5} step={0.01} onChange={setLr} color={C.red} fmt={v => v.toFixed(2)} tip={mode === "beginner" ? "How big a step the model takes each iteration. Too big = bouncy loss. Too small = very slow. 0.3-1.0 is usually good." : "Step size alpha in theta := theta - alpha * gradient. High alpha risks overshooting the minimum."} />

            {lrWarning && <div style={{ fontSize: 11, color: C.red, fontFamily: "Source Sans 3, sans-serif", marginBottom: 10, background: "#fff0f0", padding: "6px 10px", border: "1px solid " + C.red }}>Learning rate is very high -- watch the loss bounce around!</div>}
            {lrSlow && <div style={{ fontSize: 11, color: C.gold, fontFamily: "Source Sans 3, sans-serif", marginBottom: 10, background: "#fffbf0", padding: "6px 10px", border: "1px solid " + C.gold }}>Very low learning rate -- convergence will be slow.</div>}

            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: C.gray400, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8, fontFamily: "Source Sans 3, sans-serif" }}>
                {mode === "beginner" ? "Try these presets:" : "LR Presets:"}
              </div>
              <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                {[["Slow", 0.05], ["Good", 0.5], ["Fast", 1.5], ["Too High", 3.0]].map(([lbl, v]) => (
                  <button key={v} onClick={() => { setLr(v); doReset(); }} style={{ flex: 1, minWidth: 70, padding: "5px 4px", background: lr === v ? C.navy : C.offWhite, border: "1px solid " + (lr === v ? C.navy : C.gray200), color: lr === v ? C.white : C.gray600, fontSize: 10, cursor: "pointer" }}>
                    {lbl}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ background: C.offWhite, padding: "10px 12px", borderTop: "2px solid " + C.navy, marginBottom: 12 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: C.gray400, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8, fontFamily: "Source Sans 3, sans-serif" }}>Current Parameters</div>
              {[["w1 (" + ds.f1 + ")", disp.w1], ["w2 (" + ds.f2 + ")", disp.w2], ["bias b", disp.b]].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "Source Sans 3, sans-serif", marginBottom: 4 }}>
                  <span style={{ color: C.gray500 }}>{k}</span>
                  <span style={{ color: C.navy, fontWeight: 700, fontFamily: "Playfair Display, serif" }}>{v.toFixed(3)}</span>
                </div>
              ))}
              {mode === "advanced" && (
                <div style={{ marginTop: 8, fontSize: 10, color: C.gray400, fontFamily: "monospace" }}>
                  Boundary: {ds.f2} = {disp.w2 !== 0 ? ((-disp.w1 / disp.w2).toFixed(2)) : "n/a"}*{ds.f1} + {disp.w2 !== 0 ? ((-disp.b / disp.w2).toFixed(2)) : "n/a"}
                </div>
              )}
            </div>
          </Panel>

          {/* Event log */}
          <Panel>
            <STitle>Training Log</STitle>
            <div style={{ background: "#0a0c0f", padding: "10px 12px", borderRadius: 3, height: 130, overflowY: "auto", display: "flex", flexDirection: "column-reverse" }}>
              {[...log].reverse().map((entry, i) => (
                <div key={i} style={{ fontSize: 10, color: i === 0 ? "#7dffb0" : "#3d6e50", fontFamily: "monospace", lineHeight: 1.7, whiteSpace: "nowrap" }}>&gt; {entry}</div>
              ))}
            </div>
          </Panel>
        </div>
      </div>

      {/* Loss curve */}
      <Panel>
        <STitle>Loss Curve -- Training Progress</STitle>
        {mode === "beginner" && (
          <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.7, marginBottom: 12, fontFamily: "Source Sans 3, sans-serif" }}>
            A smooth downward slope = healthy training. Bouncing up and down = learning rate too high. Flat from the start = learning rate too small. Try the presets above to see the difference!
          </p>
        )}
        <ResponsiveContainer width="100%" height={190}>
          <AreaChart data={history} margin={{ top: 4, right: 16, bottom: 20, left: 0 }}>
            <defs>
              <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={C.navy} stopOpacity={0.2} />
                <stop offset="95%" stopColor={C.navy} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke={C.gray100} />
            <XAxis dataKey="step" {...axP} label={{ value: "Iteration", fill: C.gray400, fontSize: 10, position: "insideBottomRight", offset: -8 }} />
            <YAxis {...axP} tickCount={5} />
            <ReTip {...ttProps} formatter={v => [v.toFixed(5), "BCE Loss"]} />
            <Area type="monotone" dataKey="loss" stroke={C.navy} fill="url(#lossGrad)" strokeWidth={2.5} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </Panel>

      {/* Verdict card */}
      {verdict && (
        <Panel style={{ borderTop: "3px solid " + C.teal }}>
          <STitle>Model Verdict</STitle>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
            <div style={{ flex: 2, minWidth: 280, fontFamily: "Source Sans 3, sans-serif", fontSize: 13, color: C.gray600, lineHeight: 2 }}>
              <p>
                Training completed after {MAX} iterations. The model learned that <strong style={{ color: C.navy }}>{ds.f1}</strong> has weight <strong style={{ color: C.navy }}>{verdict.w1.toFixed(2)}</strong> and <strong style={{ color: C.navy }}>{ds.f2}</strong> has weight <strong style={{ color: C.navy }}>{verdict.w2.toFixed(2)}</strong>.
              </p>
              <p style={{ marginTop: 8 }}>
                {Math.abs(verdict.w1) > Math.abs(verdict.w2)
                  ? ds.f1 + " is the stronger predictor (|w1|=" + Math.abs(verdict.w1).toFixed(2) + " vs |w2|=" + Math.abs(verdict.w2).toFixed(2) + ")."
                  : ds.f2 + " is the stronger predictor (|w2|=" + Math.abs(verdict.w2).toFixed(2) + " vs |w1|=" + Math.abs(verdict.w1).toFixed(2) + ")."}
              </p>
              <p style={{ marginTop: 8 }}>
                Final performance: <strong style={{ color: C.teal }}>{(verdict.acc * 100).toFixed(1)}% accuracy</strong> and <strong style={{ color: C.teal }}>F1 = {(verdict.f1 * 100).toFixed(1)}%</strong> on {ds.data.length} samples. Navigate to Tab III to explore metrics in detail.
              </p>
              {mode === "beginner" && (
                <p style={{ marginTop: 8, fontSize: 12, color: C.gray400, fontStyle: "italic" }}>{ds.analogy}</p>
              )}
            </div>
            <div style={{ flex: 1, minWidth: 200 }}>
              {[["Accuracy", verdict.acc, C.navy], ["Precision", verdict.prec, C.teal], ["Recall", verdict.rec, C.gold], ["F1 Score", verdict.f1, C.red]].map(([n, v, c]) => (
                <div key={n} style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "Source Sans 3, sans-serif", marginBottom: 4 }}>
                    <span style={{ color: C.gray500 }}>{n}</span>
                    <span style={{ fontWeight: 700, color: c, fontFamily: "Playfair Display, serif" }}>{(v * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 5, background: C.gray100 }}>
                    <div style={{ height: "100%", width: (v * 100) + "%", background: c, transition: "width 0.5s" }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}

// ── TAB III: EVALUATE ─────────────────────────────────────────────────────────
function EvaluateTab({ mode, ds, trainedParams }) {
  const [w1, setW1] = useState(1.5);
  const [w2, setW2] = useState(0.9);
  const [b, setB] = useState(0.1);
  const [thresh, setThresh] = useState(0.5);
  const [fpCost, setFpCost] = useState(1);
  const [fnCost, setFnCost] = useState(1);

  useEffect(() => {
    if (trainedParams) { setW1(trainedParams.w1); setW2(trainedParams.w2); setB(trainedParams.b); }
  }, [trainedParams]);

  const m = calcMetrics(w1, w2, b, ds.data, thresh);
  const roc = calcRoc(w1, w2, b, ds.data);
  const pr = calcPr(w1, w2, b, ds.data);
  const auc = calcAuc(roc).toFixed(3);
  const aucLabel = +auc >= 0.9 ? "Excellent" : +auc >= 0.8 ? "Good" : +auc >= 0.7 ? "Fair" : "Poor";
  const aucColor = +auc >= 0.9 ? C.teal : +auc >= 0.8 ? C.navy : +auc >= 0.7 ? C.gold : C.red;
  const diag = Array.from({ length: 11 }, (_, i) => ({ fpr: i / 10, tpr: i / 10 }));

  const optThresh = (() => {
    let best = 0.5, bestCost = Infinity;
    for (let t = 0.05; t <= 0.95; t += 0.05) {
      const ev = calcMetrics(w1, w2, b, ds.data, +t.toFixed(2));
      const cost = ev.fp * fpCost + ev.fn * fnCost;
      if (cost < bestCost) { bestCost = cost; best = t; }
    }
    return best.toFixed(2);
  })();

  const totalCost = m.fp * fpCost + m.fn * fnCost;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
        <KpiCard label="Accuracy" value={(m.acc * 100).toFixed(1) + "%"} sub="(TP+TN) / N" accent={C.navy} tip={mode === "beginner" ? "Out of all predictions, what fraction were correct?" : "(TP+TN)/N. Can be misleading with imbalanced classes."} />
        <KpiCard label="Precision" value={(m.prec * 100).toFixed(1) + "%"} sub="TP / (TP+FP)" accent={C.teal} tip={mode === "beginner" ? "Of everything the model called positive, what fraction actually was?" : "TP/(TP+FP). High precision = few false alarms."} />
        <KpiCard label="Recall" value={(m.rec * 100).toFixed(1) + "%"} sub="TP / (TP+FN)" accent={C.gold} tip={mode === "beginner" ? "Of all actual positives, what fraction did the model catch?" : "TP/(TP+FN). High recall = few missed positives. Also called sensitivity or TPR."} />
        <KpiCard label="F1 Score" value={(m.f1 * 100).toFixed(1) + "%"} sub="2*P*R / (P+R)" accent={C.red} tip={mode === "beginner" ? "A single score balancing precision and recall. High F1 = both accurate and thorough." : "Harmonic mean of precision and recall. More robust than accuracy for imbalanced classes."} />
        <KpiCard label={"AUC -- " + aucLabel} value={auc} sub="Area under ROC" accent={aucColor} tip={mode === "beginner" ? "Pick one true positive and one true negative at random. AUC = chance the model ranks the positive higher. 1.0 = perfect." : "P(score_pos > score_neg). Measures discrimination power across all thresholds."} />
      </div>

      {/* Parameter controls */}
      <Panel>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
          <div style={{ flex: 2, minWidth: 280 }}>
            <STitle>Model Parameters</STitle>
            {trainedParams && (
              <div style={{ fontSize: 11, color: C.teal, fontFamily: "Source Sans 3, sans-serif", marginBottom: 12, background: "#e8f5e9", padding: "7px 10px", border: "1px solid " + C.teal }}>
                Using weights trained in Tab II. Adjust sliders to explore what-ifs.
              </div>
            )}
            <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
              <div style={{ flex: 1, minWidth: 160 }}>
                <SliderRow label={"w1 (" + ds.f1 + ")"} value={w1} min={-3} max={3} step={0.1} onChange={setW1} color={C.navy} />
                <SliderRow label={"w2 (" + ds.f2 + ")"} value={w2} min={-3} max={3} step={0.1} onChange={setW2} color={C.teal} />
              </div>
              <div style={{ flex: 1, minWidth: 160 }}>
                <SliderRow label="Bias (b)" value={b} min={-3} max={3} step={0.1} onChange={setB} color={C.navyMid} />
                <SliderRow label="Threshold" value={thresh} min={0.05} max={0.95} step={0.05} onChange={setThresh} color={C.red} fmt={v => v.toFixed(2)} tip={mode === "beginner" ? "How sure does the model need to be before predicting positive? Lower = catch more, but more false alarms." : "Classification cutoff tau. Increases recall at expense of precision when lowered."} />
              </div>
            </div>
          </div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <STitle>Threshold Consequences</STitle>
            <div style={{ background: C.offWhite, padding: "12px 14px", border: "1px solid " + C.gray200, fontFamily: "Source Sans 3, sans-serif", fontSize: 12, lineHeight: 2, color: C.gray600 }}>
              <div>At threshold = <strong style={{ color: C.red }}>{thresh.toFixed(2)}</strong>:</div>
              <div>Catching <strong style={{ color: C.teal }}>{(m.rec * 100).toFixed(0)}%</strong> of {ds.cls1} cases</div>
              <div>False alarms: <strong style={{ color: m.fp > 8 ? C.red : C.gold }}>{m.fp}</strong> of {m.fp + m.tn} {ds.cls0} cases</div>
              <div>Missed: <strong style={{ color: m.fn > 8 ? C.red : C.gold }}>{m.fn}</strong> of {m.tp + m.fn} {ds.cls1} cases</div>
              {mode === "beginner" && (
                <div style={{ marginTop: 6, fontSize: 11, color: C.gray400, fontStyle: "italic" }}>
                  {thresh < 0.3 ? "Very low: catching almost everything but many false alarms." : thresh > 0.7 ? "Very high: very precise but missing many positives." : "Balanced: reasonable precision-recall tradeoff."}
                </div>
              )}
            </div>
          </div>
        </div>
      </Panel>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {/* Confusion matrix */}
        <Panel style={{ flex: 1, minWidth: 280 }}>
          <STitle>Confusion Matrix</STitle>
          {mode === "beginner" && (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.6, marginBottom: 14, fontFamily: "Source Sans 3, sans-serif" }}>
              The four possible outcomes. You want the diagonal (TN + TP) to be large and the off-diagonal (FP + FN) to be small.
            </p>
          )}
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 1, background: C.gray200 }}>
            <div style={{ background: C.white, padding: 8 }} />
            {["Actual " + ds.cls0, "Actual " + ds.cls1].map(h => (
              <div key={h} style={{ background: C.offWhite, textAlign: "center", padding: "7px 4px", fontSize: 10, fontFamily: "Source Sans 3, sans-serif", fontWeight: 600, color: C.gray500, textTransform: "uppercase" }}>{h}</div>
            ))}
            {[
              { row: "Pred " + ds.cls0, cells: [{ v: m.tn, label: "TN", c: C.teal, desc: "Correct no" }, { v: m.fn, label: "FN", c: C.gold, desc: "Missed yes" }] },
              { row: "Pred " + ds.cls1, cells: [{ v: m.fp, label: "FP", c: C.red, desc: "False alarm" }, { v: m.tp, label: "TP", c: C.navy, desc: "Correct yes" }] },
            ].map(({ row, cells }) => (
              <>
                <div key={row} style={{ background: C.offWhite, padding: "8px 10px", fontSize: 10, fontFamily: "Source Sans 3, sans-serif", fontWeight: 600, color: C.gray500, textTransform: "uppercase", display: "flex", alignItems: "center" }}>{row}</div>
                {cells.map(({ v, label, c, desc }) => (
                  <div key={label} style={{ background: C.white, padding: "16px 10px", textAlign: "center", borderTop: "3px solid " + c }}>
                    <div style={{ fontSize: 26, fontFamily: "Playfair Display, serif", fontWeight: 700, color: c }}>{v}</div>
                    <div style={{ fontSize: 9, color: c, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", fontFamily: "Source Sans 3, sans-serif", marginTop: 3 }}>{label}</div>
                    {mode === "beginner" && <div style={{ fontSize: 9, color: C.gray400, fontFamily: "Source Sans 3, sans-serif", marginTop: 2 }}>{desc}</div>}
                  </div>
                ))}
              </>
            ))}
          </div>
        </Panel>

        {/* Metrics bars + cost simulator */}
        <Panel style={{ flex: 1, minWidth: 260 }}>
          <STitle>Metric Breakdown</STitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 18 }}>
            {[
              { name: "Accuracy", val: m.acc, color: C.navy, formula: "(TP+TN)/N" },
              { name: "Precision", val: m.prec, color: C.teal, formula: "TP/(TP+FP)" },
              { name: "Recall", val: m.rec, color: C.gold, formula: "TP/(TP+FN)" },
              { name: "F1 Score", val: m.f1, color: C.red, formula: "2*P*R/(P+R)" },
            ].map(({ name, val, color, formula }) => (
              <div key={name}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span className="tip" style={{ fontSize: 12, fontWeight: 600, color: C.navy, fontFamily: "Source Sans 3, sans-serif" }}>
                    {name}
                    {mode === "advanced" && <span className="tipbox">{formula}</span>}
                  </span>
                  <span style={{ fontSize: 14, fontFamily: "Playfair Display, serif", fontWeight: 700, color }}>{(val * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: 6, background: C.gray100, borderRadius: 1 }}>
                  <div style={{ height: "100%", width: (val * 100) + "%", background: color, borderRadius: 1, transition: "width 0.35s ease" }} />
                </div>
                {mode === "advanced" && <div style={{ fontSize: 10, color: C.gray400, fontFamily: "monospace", marginTop: 3 }}>{formula}</div>}
              </div>
            ))}
          </div>

          <div style={{ borderTop: "1px solid " + C.gray200, paddingTop: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: C.gray400, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8, fontFamily: "Source Sans 3, sans-serif" }}>
              {mode === "beginner" ? "Business Cost Simulator" : "Cost-Benefit Analysis"}
            </div>
            {mode === "beginner" && <p style={{ fontSize: 11, color: C.gray500, marginBottom: 10, fontFamily: "Source Sans 3, sans-serif", lineHeight: 1.5 }}>Different use cases have different error costs. In fraud detection, a missed fraud (FN) is far more costly than a false alarm (FP).</p>}
            <SliderRow label={"FP cost (false " + ds.cls1 + ")"} value={fpCost} min={1} max={10} step={1} onChange={setFpCost} color={C.red} fmt={v => "$" + v} />
            <SliderRow label={"FN cost (missed " + ds.cls1 + ")"} value={fnCost} min={1} max={10} step={1} onChange={setFnCost} color={C.gold} fmt={v => "$" + v} />
            <div style={{ background: C.offWhite, padding: "10px 12px", border: "1px solid " + C.gray200, fontSize: 12, fontFamily: "Source Sans 3, sans-serif" }}>
              <div style={{ color: C.navy }}>Optimal threshold: <strong>{optThresh}</strong></div>
              <div style={{ color: C.gray500, marginTop: 3, fontSize: 11 }}>Current total cost at tau={thresh.toFixed(2)}: <strong style={{ color: C.red }}>${totalCost}</strong></div>
            </div>
          </div>
        </Panel>

        {/* ROC + PR curves */}
        <Panel style={{ flex: 2, minWidth: 320 }}>
          <STitle>ROC and Precision-Recall Curves</STitle>
          {mode === "beginner" ? (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.6, marginBottom: 14, fontFamily: "Source Sans 3, sans-serif" }}>
              <strong style={{ color: C.navy }}>ROC curve:</strong> Model vs. coin flip. Closer to top-left corner = better. AUC = {auc} ({aucLabel}).<br />
              <strong style={{ color: C.teal }}>PR curve:</strong> Use this when positives are rare (fraud, disease). Higher and further right = better.
            </p>
          ) : (
            <p style={{ fontSize: 12, color: C.gray500, lineHeight: 1.6, marginBottom: 14, fontFamily: "Source Sans 3, sans-serif" }}>
              ROC: TPR vs FPR at all thresholds. AUC = P(score_pos &gt; score_neg) = {auc}. PR curve preferred for imbalanced classes -- ROC can be optimistic when negatives vastly outnumber positives.
            </p>
          )}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 200 }}>
              <div style={{ fontSize: 10, color: C.gray400, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, fontFamily: "Source Sans 3, sans-serif" }}>ROC -- AUC = {auc} ({aucLabel})</div>
              <ResponsiveContainer width="100%" height={190}>
                <LineChart margin={{ top: 4, right: 8, bottom: 26, left: 0 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke={C.gray100} />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...axP} tickCount={5} label={{ value: "FPR", fill: C.gray400, fontSize: 9, position: "insideBottom", offset: -14 }} />
                  <YAxis dataKey="tpr" type="number" domain={[0, 1]} {...axP} tickCount={5} label={{ value: "TPR", fill: C.gray400, fontSize: 9, angle: -90, position: "insideLeft", offset: 8 }} />
                  <ReTip {...ttProps} formatter={(v, n) => [v.toFixed(3), n]} />
                  <Line data={diag} type="linear" dataKey="tpr" stroke={C.gray300} strokeWidth={1} strokeDasharray="5 5" dot={false} name="Random" />
                  <Line data={roc} type="monotone" dataKey="tpr" stroke={C.navy} strokeWidth={2.5} dot={false} name="Model" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{ flex: 1, minWidth: 200 }}>
              <div style={{ fontSize: 10, color: C.gray400, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8, fontFamily: "Source Sans 3, sans-serif" }}>Precision-Recall Curve</div>
              <ResponsiveContainer width="100%" height={190}>
                <LineChart margin={{ top: 4, right: 8, bottom: 26, left: 0 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke={C.gray100} />
                  <XAxis dataKey="rec" type="number" domain={[0, 1]} {...axP} tickCount={5} label={{ value: "Recall", fill: C.gray400, fontSize: 9, position: "insideBottom", offset: -14 }} />
                  <YAxis dataKey="prec" type="number" domain={[0, 1]} {...axP} tickCount={5} label={{ value: "Precision", fill: C.gray400, fontSize: 9, angle: -90, position: "insideLeft", offset: 8 }} />
                  <ReTip {...ttProps} formatter={(v, n) => [v.toFixed(3), n]} />
                  <Line data={pr} type="monotone" dataKey="prec" stroke={C.teal} strokeWidth={2.5} dot={false} name="Model" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            {[["0.9-1.0", "Excellent", C.teal], ["0.8-0.9", "Good", C.navy], ["0.7-0.8", "Fair", C.gold], ["< 0.7", "Poor", C.red]].map(([range, label, color]) => (
              <div key={range} style={{ flex: 1, minWidth: 80, borderLeft: "3px solid " + color, paddingLeft: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color, fontFamily: "Source Sans 3, sans-serif" }}>{label}</div>
                <div style={{ fontSize: 10, color: C.gray400, fontFamily: "Source Sans 3, sans-serif" }}>{range}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

// ── APP SHELL ─────────────────────────────────────────────────────────────────
const TABS = [
  { id: "understand", label: "I. Understand" },
  { id: "train",      label: "II. Train" },
  { id: "evaluate",   label: "III. Evaluate" },
];

function LogisticVisualizer()  {
  const [tab, setTab] = useState("understand");
  const [mode, setMode] = useState("beginner");
  const [dsKey, setDsKey] = useState("exam");
  const [trainedParams, setTrainedParams] = useState(null);

  const ds = DATASETS[dsKey];

  const handleDatasetChange = (key) => {
    setDsKey(key);
    setTrainedParams(null);
  };

  return (
    <>
      <style>{CSS}</style>
      <div style={{ background: C.offWhite, minHeight: "100vh", fontFamily: "Source Sans 3, sans-serif" }}>

        {/* Header */}
        <div style={{ background: C.navy, padding: "0 36px" }}>
          <div style={{ maxWidth: 1280, margin: "0 auto" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12, paddingTop: 18 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                <div style={{ width: 4, height: 34, background: C.red }} />
                <div>
                  <h1 style={{ fontFamily: "Playfair Display, serif", fontSize: 21, fontWeight: 700, color: C.white }}>Logistic Regression</h1>
                  <div style={{ fontSize: 10, color: "#7a9ab8", letterSpacing: "0.12em", textTransform: "uppercase", marginTop: 2, fontWeight: 600 }}>World-Class Interactive Learning Tool</div>
                </div>
              </div>

              <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
                {/* Dataset picker */}
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 10, color: "#7a9ab8", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600, fontFamily: "Source Sans 3, sans-serif" }}>Dataset:</span>
                  <select value={dsKey} onChange={e => handleDatasetChange(e.target.value)}>
                    {Object.entries(DATASETS).map(([k, v]) => (
                      <option key={k} value={k}>{v.label}</option>
                    ))}
                  </select>
                </div>

                {/* Mode toggle */}
                <div style={{ display: "flex", background: "rgba(255,255,255,0.1)", borderRadius: 4, padding: 2, gap: 2 }}>
                  {["beginner", "advanced"].map(m => (
                    <button key={m} onClick={() => setMode(m)} style={{ padding: "5px 14px", background: mode === m ? C.white : "transparent", border: "none", color: mode === m ? C.navy : "#7a9ab8", fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", cursor: "pointer", borderRadius: 3, transition: "all 0.15s" }}>
                      {m === "beginner" ? "Beginner" : "Advanced"}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Tab bar */}
            <div style={{ display: "flex", alignItems: "flex-end", gap: 2, marginTop: 10 }}>
              {TABS.map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} style={{ padding: "12px 22px", background: tab === t.id ? C.white : "transparent", border: "none", borderTop: tab === t.id ? "3px solid " + C.red : "3px solid transparent", color: tab === t.id ? C.navy : "#7a9ab8", fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer", transition: "all 0.2s", whiteSpace: "nowrap" }}>
                  {t.label}
                </button>
              ))}
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "flex-end", paddingBottom: 8 }}>
                <div style={{ background: "rgba(255,255,255,0.08)", padding: "4px 12px", fontSize: 10, color: "#7a9ab8", fontFamily: "Source Sans 3, sans-serif" }}>
                  {ds.label} -- {ds.useCase} -- {mode} mode
                  {trainedParams && <span style={{ color: "#7dffb0", marginLeft: 8 }}>Model trained</span>}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Body */}
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 36px" }}>
          {tab === "understand" && <UnderstandTab mode={mode} ds={ds} />}
          {tab === "train"      && <TrainTab mode={mode} ds={ds} onTrained={setTrainedParams} />}
          {tab === "evaluate"   && <EvaluateTab mode={mode} ds={ds} trainedParams={trainedParams} />}
        </div>

        {/* Footer */}
        <div style={{ borderTop: "1px solid " + C.gray200, padding: "14px 36px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <div style={{ fontSize: 10, color: C.gray400, letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 600, fontFamily: "Source Sans 3, sans-serif" }}>
            Logistic Regression -- Interactive Learning Tool
          </div>
          <div style={{ fontSize: 10, color: C.gray300, fontFamily: "Source Sans 3, sans-serif" }}>
            {ds.data.length} data points -- Gradient descent -- Binary cross-entropy -- {mode} mode
          </div>
        </div>
      </div>
    </>
  );
}

export default LogisticVisualizer;