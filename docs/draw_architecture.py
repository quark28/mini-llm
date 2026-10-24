"""Paper-style architecture figure for the masked-diffusion encoder-decoder LM."""
import cairosvg, os, sys

OUT = sys.argv[1]
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "thinking"   # "thinking" | "classic"
THINK = VARIANT == "thinking"
os.makedirs(OUT, exist_ok=True)

W, H = 840, 800
FONT = "Helvetica, Arial, 'DejaVu Sans', sans-serif"

C = {
    "attn": ("#FCE3CF", "#D9895B"),
    "ffn": ("#D6E6F7", "#5B8BC9"),
    "norm": ("#F6F0C4", "#B8A63E"),
    "tied": ("#DDF0DD", "#5FA36A"),
    "think": ("#EBE1F5", "#8A6BB8"),
    "loss": ("#EFEFEF", "#8A8A8A"),
}

parts = []


def add(s):
    parts.append(s)


def text(x, y, s, size=13, weight="normal", anchor="middle", color="#222", style="normal"):
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" text-anchor="{anchor}" fill="{color}">{s}</text>')


def box(x0, y0, x1, y1, kind, lines, r=6):
    fill, stroke = C[kind]
    add(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    cx = (x0 + x1) / 2
    n = len(lines)
    total = 15 + (n - 1) * 13
    y = (y0 + y1) / 2 - total / 2 + 11
    for i, ln in enumerate(lines):
        if i == 0:
            text(cx, y, ln, 13, "bold")
            y += 15
        else:
            text(cx, y, ln, 11, color="#444")
            y += 13


def group(x0, y0, x1, y1, fill="#FAFAFA", stroke="#9A9A9A", dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ""
    add(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.3"{d}/>')


def arrow(pts, color="#333", width=1.4, dash=False, head=True):
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    m = ' marker-end="url(#ah)"' if head else ""
    da = ' stroke-dasharray="5 4"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{m}{da}/>')


def plus(cx, cy, r=10):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="#333" stroke-width="1.4"/>')
    add(f'<line x1="{cx-6}" y1="{cy}" x2="{cx+6}" y2="{cy}" stroke="#333" stroke-width="1.4"/>')
    add(f'<line x1="{cx}" y1="{cy-6}" x2="{cx}" y2="{cy+6}" stroke="#333" stroke-width="1.4"/>')


def sublayer(cx, x0, x1, rx, norm, sub, plus_y, sub_kind, sub_lines, start_y):
    """Pre-norm residual sub-layer, drawn bottom-up.
    start_y: y where the residual stream enters from below."""
    ny0, ny1 = norm
    sy0, sy1 = sub
    branch = ny1 + 10
    arrow([(cx, start_y), (cx, ny1)])
    add(f'<circle cx="{cx}" cy="{branch}" r="2.6" fill="#333"/>')
    box(x0, ny0, x1, ny1, "norm", ["RMSNorm"])
    arrow([(cx, ny0), (cx, sy1)])
    box(x0, sy0, x1, sy1, sub_kind, sub_lines)
    arrow([(cx, sy0), (cx, plus_y + 10)])
    plus(cx, plus_y)
    side = 10 if rx > cx else -10
    arrow([(cx, branch), (rx, branch), (rx, plus_y), (cx + side, plus_y)])
    return plus_y - 10


# ---------------------------------------------------------------- header
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---------------------------------------------------------------- encoder column
ecx, ex0, ex1, erx = 180, 80, 280, 296
group(50, 365, 312, 632)
text(34, 505, "×4" if THINK else "×6", 18, "bold")
text(181, 727, "Context tokens   [B, 256]", 13, "bold")
text(181, 745, "clean, left-padded dialogue history", 11, color="#555")
box(ex0, 650, ex1, 684, "tied", ["Token Embedding  E"])
arrow([(ecx, 712), (ecx, 684)])
y = sublayer(ecx, ex0, ex1, erx, (585, 611), (525, 569), 500, "attn",
             ["Multi-Head Self-Attention", "bidirectional · RoPE · pad mask"], 650)
y = sublayer(ecx, ex0, ex1, erx, (455, 481), (398, 436), 380, "ffn",
             ["SwiGLU FFN"], y)
text(ecx, 360 + 0, "", 1)
text(64, 382, "Encoder", 12, "bold", anchor="start", color="#666")
arrow([(ecx, y), (ecx, 346)])
box(ex0, 320, ex1, 346, "norm", ["RMSNorm"])

# ---------------------------------------------------------------- Thinking module
if THINK:
    group(50, 92, 312, 302, fill="#F8F4FC", stroke="#8A6BB8", dash=True)
    text(64, 110, "Thinking module", 12, "bold", anchor="start", color="#6A4C98")
    arrow([(ecx, 320), (ecx, 290)])
    box(80, 256, 280, 290, "tied", ["Cosine energy vs. vocabulary", "s = cos(H, E)   256 × 12 288"])
    # two channels
    arrow([(130, 256), (130, 241)])
    arrow([(230, 256), (230, 241)])
    box(66, 181, 176, 241, "think", ["Words", "per position", "α-weighted  s · E"])
    box(186, 181, 298, 241, "think", ["Thought", "Σ positions, centred", "LN → SiLU → proj"])
    plus(ecx, 162)
    arrow([(130, 181), (130, 162), (ecx - 10, 162)])
    arrow([(230, 181), (230, 162), (ecx + 10, 162)])
    arrow([(ecx, 152), (ecx, 146)])
    box(ex0, 120, ex1, 146, "norm", ["RMSNorm"])

# memory route to cross-attention
if THINK:
    mem_y = 133
    text(432, 300, "memory  M", 12, "bold", anchor="end", color="#6A4C98")
    text(432, 315, "[B, 256, 384]", 11, anchor="end", color="#6A4C98")
    text(432, 332, "computed once", 11, anchor="end", color="#666", style="italic")
    text(432, 345, "per message", 11, anchor="end", color="#666", style="italic")
else:
    mem_y = 333
    text(300, 282, "memory  M", 12, "bold", anchor="start", color="#6A4C98")
    text(300, 296, "[B, 256, 384]", 11, anchor="start", color="#6A4C98")
    text(300, 311, "computed once", 11, anchor="start", color="#666", style="italic")
    text(300, 324, "per message", 11, anchor="start", color="#666", style="italic")

# ---------------------------------------------------------------- decoder column
dcx, dx0, dx1, drx = 620, 520, 720, 736
group(490, 240, 752, 632)
text(766, 440, "×2", 18, "bold", anchor="start")
text(504, 257, "Decoder", 12, "bold", anchor="start", color="#666")
text(620, 727, "Noisy answer tokens   [B, L_ans]", 13, "bold")
text(620, 745, "each token → [MASK] with prob. t,  t ~ U(0, 1)", 11, color="#555")
box(dx0, 650, dx1, 684, "tied", ["Token Embedding  E"])
arrow([(dcx, 712), (dcx, 684)])
y = sublayer(dcx, dx0, dx1, drx, (585, 611), (525, 569), 500, "attn",
             ["Multi-Head Self-Attention", "bidirectional · RoPE"], 650)
y = sublayer(dcx, dx0, dx1, drx, (455, 481), (395, 439), 370, "attn",
             ["Multi-Head Cross-Attention", "Q: answer · K, V: memory"], y)
y = sublayer(dcx, dx0, dx1, drx, (325, 351), (275, 311), 255, "ffn",
             ["SwiGLU FFN"], y)
arrow([(dcx, y), (dcx, 221)])
box(dx0, 195, dx1, 221, "norm", ["RMSNorm"])
arrow([(dcx, 195), (dcx, 174)])
box(dx0, 136, dx1, 174, "tied", ["LM head", "weights tied to E"])
arrow([(dcx, 140), (dcx, 104)])
text(dcx + 8, 125, "logits  [B, L_ans, 12 288]", 11, anchor="start", color="#444")
box(500, 52, 740, 104, "loss", ["Masked-diffusion objective", "CE on masked positions, weighted 1/t", "(pretraining and SFT share the same step)"])

arrow([(ex1, mem_y), (440, mem_y), (440, 417), (dx0, 417)], width=1.6)

# ---------------------------------------------------------------- legend
lx, ly = 380, 770
items = [("attn", "attention"), ("ffn", "feed-forward"), ("norm", "normalisation"), ("tied", "weight-tied to E")]
if THINK:
    items.append(("think", "Thinking module"))
step = 158 if THINK else 200
for i, (k, lab) in enumerate(items):
    xx = (40 if THINK else 80) + i * step
    f, s = C[k]
    add(f'<rect x="{xx}" y="{ly-11}" width="16" height="13" rx="3" fill="{f}" stroke="{s}" stroke-width="1.2"/>')
    text(xx + 22, ly, lab, 11, anchor="start", color="#444")
text(W / 2, 22, "", 1)

add("</svg>")
svg = "\n".join(parts)
with open(os.path.join(OUT, f"architecture-{VARIANT}.svg"), "w") as fh:
    fh.write(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to=os.path.join(OUT, f"architecture-{VARIANT}.png"), scale=2.0)
print("done")
