# -*- coding: utf-8 -*-
"""SWoPP/CPSY 発表スライド構成書 (.docx) を生成するスクリプト。"""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# --- 既定フォント（日本語対応）---
style = doc.styles["Normal"]
style.font.name = "游ゴシック"
style.font.size = Pt(10.5)
# 東アジアフォント指定
from docx.oxml.ns import qn
style.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)  # 濃紺


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = ACCENT
        run.font.name = "游ゴシック"
        run.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
    return p


def para(text, bold=False, italic=False, size=10.5, space_after=4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    r.font.name = "游ゴシック"
    r.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    if level:
        p.paragraph_format.left_indent = Pt(18 * (level + 1))
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.name = "游ゴシック"
    r.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
    return p


def kv(label, value):
    """『ラベル：値』の1行。"""
    p = doc.add_paragraph()
    r1 = p.add_run(label + "：")
    r1.bold = True
    r2 = p.add_run(value)
    for r in (r1, r2):
        r.font.size = Pt(10.5)
        r.font.name = "游ゴシック"
        r.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
    p.paragraph_format.space_after = Pt(3)
    return p


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(htext)
        run.bold = True
        run.font.size = Pt(10)
        run.font.name = "游ゴシック"
        run.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(val)
            run.font.size = Pt(9.5)
            run.font.name = "游ゴシック"
            run.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def slide(num, title, minutes, material):
    h(f"Slide {num} — {title}", level=2)
    meta = doc.add_paragraph()
    r = meta.add_run(f"［目安 {minutes}／素材：{material}］")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    r.font.name = "游ゴシック"
    r.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
    meta.paragraph_format.space_after = Pt(3)


# =========================================================
# タイトル
# =========================================================
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title_p.add_run("SWoPP 2026 / CPSY 発表スライド構成書")
r.bold = True
r.font.size = Pt(18)
r.font.color.rgb = ACCENT
r.font.name = "游ゴシック"
r.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("「アクティブトークン更新による MaskGIT の推論効率化」\n本編17枚＋バックアップ5枚／発表20分・質疑5分想定")
r.font.size = Pt(11)
r.font.name = "游ゴシック"
r.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")

doc.add_paragraph()

# --- 全体ルール ---
h("全体ルール（毎ページ共通）", level=1)
bullet("画面比 16:9。本文フォント 24pt 以上、見出し 32pt 以上（スライド上での話）。")
bullet("配色：ベース白＋アクセント1色（濃紺）。赤は U_t（現ステップ確定）と警告のみに限定。")
bullet("1枚1メッセージ。見出しは体言止めではなく「主張の一文」にする。")
bullet("右下にページ番号、左下に「SWoPP 2026 CPSY／リシュウカン（東京科学大学）」。")
bullet("図は論文の該当図を流用可。図中英語ラベルは残してよいが、キャプションは日本語にする。")

doc.add_page_break()

# =========================================================
# 本編スライド
# =========================================================
h("本編スライド（17枚）", level=1)

# 1
slide(1, "タイトル", "0:30", "―")
kv("大見出し", "アクティブトークン更新による MaskGIT の推論効率化")
kv("副題", "― Halton スケジューラの事前決定性を利用した FFN 出力キャッシュ ―")
kv("著者", "リシュウカン，市川雄樹，本村真人，藤木大地，金子竜也（東京科学大学）")
kv("会議", "SWoPP 2026 / 情報処理学会 計算機システム研究会（CPSY）")
para("口頭：一言の自己紹介＋「本発表は反復画像生成の推論をキャッシュで高速化する話」と、system 系聴衆向けに位置づけを一文で。", italic=True)

# 2
slide(2, "背景：反復生成は推論コストが高い", "1:00", "step数比較の簡易棒グラフ")
kv("見出し", "反復生成モデルは推論ステップ数がボトルネック")
bullet("拡散モデル（DDPM）：生成に約1000ステップの逐次ノイズ除去。")
bullet("自己回帰モデル：トークン数に比例した逐次順伝播。")
bullet("MaskGIT：離散トークン列の並列復号で数十ステップに短縮。")
kv("強調", "→ ステップ数は減ったが、1ステップあたりの計算はまだ重い")
para("口頭：MaskGIT の『少step で速い』立ち位置を確認 → 本発表はその『1step の中身』を削る話、と予告。", italic=True)

# 3
slide(3, "Halton-MaskGIT：更新位置が事前に決まる", "1:00", "復号の模式図")
kv("見出し", "Halton-MaskGIT：更新トークン位置 U_t を推論前に決定")
bullet("従来 MaskGIT：予測信頼度で毎ステップ更新位置を決定。")
bullet("Halton：系列の準一様性で、更新位置をあらかじめ空間全体に配置。")
bullet("決定的にもかかわらず ImageNet で信頼度ベースを上回る品質 [Besnier+ ICLR2025]。")
kv("強調枠", "重要：更新される位置 U_t が『推論開始前に既知』")
para("口頭：この『事前既知』が後で効く、と伏線を強く張る。提案の土台。", italic=True)

# 4
slide(4, "課題：位置は既知なのに全部を再計算している", "1:00", "全577トークン×全24層の図")
kv("見出し", "更新位置は既知 — なのに毎ステップ Transformer 全体を全トークンで再計算")
bullet("各ステップで新たに確定するトークンは全体のごく一部。")
bullet("確定済み／マスク済みトークンの中間表現はステップ間でほぼ不変。")
bullet("→ これらへの計算の多くは冗長。")
kv("問い", "この冗長計算をキャッシュで省けるか？")
para("口頭：ここで『今日の一言』を言い切る＝冗長な再計算をキャッシュ再利用で削る。", italic=True)

# 5
slide(5, "どこが重いか：FFN が支配的", "1:00", "図1(a)(b)")
kv("見出し", "計算コストは Transformer 層、その内 FFN に集中")
kv("図キャプション", "図1：H-MaskGIT-L の1順伝播あたり FLOPs 内訳（d=1024, L=24, 系列長577）")
kv("数字強調", "Transformer 層 95.2% ／ その内 FFN 61.7%・注意機構 38.3%")
kv("結論", "→ FFN を削れば効果が最大 → 本研究は FFN を対象")
para("口頭：CPSY 向けに『どこがホットスポットか』を FLOPs で定量化した、という system の作法を強調。", italic=True)

# 6
slide(6, "関連研究と本研究の位置づけ（★差別化スライド）", "1:30", "比較表（新規作成）")
kv("見出し", "既存のキャッシュ高速化との違い：更新位置の『決定コスト』がゼロ")
table(
    ["手法", "キャッシュ対象", "更新/再利用位置の決定", "追加学習"],
    [
        ["DeepCache [14]", "層特徴（拡散）", "ヒューリスティック", "不要"],
        ["Learning-to-Cache [15]", "層（DiT）", "学習で選択", "必要"],
        ["LazyMAR [16]", "トークン特徴（MAR）", "実行時判定", "不要"],
        ["提案手法", "FFN 出力（トークン単位）", "事前決定（Halton）", "不要"],
    ],
)
kv("強調", "信頼度ベース MaskGIT では更新位置特定に全トークン推論が必要 → キャッシュが原理的に困難")
para("口頭：本発表の一番の売り。『追加学習不要』×『決定コストゼロ』が揃うのは提案だけ、と言い切る。", italic=True)

# 7
slide(7, "観察①：非activeトークンの FFN 出力は安定", "1:30", "図3（cos / drift ヒートマップ）")
kv("見出し", "非アクティブトークンの FFN 出力はステップ間でほぼ不変")
kv("図キャプション", "図3：非active トークンの FFN 出力差分の安定性（層×ステップ）。左=方向、右=大きさ")
bullet("中間層・中盤ステップ以降：高コサイン・低ドリフト＝安定。")
bullet("第0層付近・復号初期：不安定。")
bullet("補足：マスクトークンでも同様に成立（詳細はバックアップ）。")
para("口頭：削っていい根拠の実測。図4は出さず一文で触れる。", italic=True)

# 8
slide(8, "観察②：トークン齢で active を定義", "1:30", "図5（齢1赤・齢2以上青）")
kv("見出し", "齢1は再計算が必要、齢2以上はキャッシュ可 → A_t = U_{t-1} ∪ U_t")
bullet("齢1（直前ステップ確定 U_{t-1}）：差分が大きく変化 → 再計算。")
bullet("齢2以上：安定 → キャッシュ再利用。")
kv("定義枠", "アクティブトークン A_t = U_{t-1} ∪ U_t（現＋直前ステップで確定）")
para("口頭：なぜ『1個前まで』含めるのかを図で正当化。提案の核心設計。", italic=True)

# 9
slide(9, "提案手法：全体像（★3段アニメ）", "2:00", "図2 を3クリックで積み上げ")
kv("見出し", "提案手法の全体像：トークン分類 → 層ゲート → 層内部")
bullet("段1：トークン分類のみ表示（U_t 赤 / U_{t-1} 緑 / inactive 灰）。")
bullet("段2：層方向ゲートを重ねる（Layer 1-2 と最終層は full、中間層は Reuse cached）。")
bullet("段3：層内部を重ねる（SwiGLU が active のみ計算、inactive は Cache reuse → 残差加算）。")
para("口頭：発表の中心。粒度を1段ずつ上げて説明する。", italic=True)

# 10
slide(10, "FFN の定式化", "1:00", "数式1つ")
kv("見出し", "アクティブトークンのみ FFN 計算、非アクティブはキャッシュ加算")
para("x^{l+1}_i(t) = x^l_i(t) + FFN_l(x^l_i(t))   if i ∈ A_t", bold=True)
para("x^{l+1}_i(t) = x^l_i(t) + Cache^l_i           if i ∉ A_t", bold=True)
bullet("active：FFN を計算し、その出力でキャッシュを更新。")
bullet("inactive：計算せず、保持キャッシュを残差に加算。")
bullet("U_t が事前既知なので、計算すべき位置を実行前に特定可能。")
para("口頭：式は1つだけ。左辺右辺を指しながら30秒で。", italic=True)

# 11
slide(11, "適用範囲の制御：ステップゲート・層ゲート", "1:00", "step軸／layer軸のバンド図")
kv("見出し", "安定領域にのみ適用 — 不安定な初期ステップ・端の層は全計算")
bullet("ステップゲート：5 ≤ t < 31 に適用（t<5 と t=31 は全トークン）。")
bullet("層ゲート：中間層のみ（L=第4–22層 / B・S=第3–9層）。")
bullet("根拠は観察①（初期・第0層・上位層が不安定）。")
para("口頭：ゲート＝観察①への対応、と紐づける。層番号の基準は L/B/S で統一しておく。", italic=True)

# 12
slide(12, "周期的リフレッシュ N（品質と速度のノブ）", "1:30", "帯図＋両端の解釈")
kv("見出し", "周期的リフレッシュ：N ステップに1度だけ全トークン再計算")
bullet("キャッシュを使い続けると誤差が蓄積 → 定期的に全計算でリセット。")
bullet("N=1：毎回全計算 ＝ ベースライン。")
bullet("N→∞：リフレッシュ無し ＝ 純粋な部分更新。")
bullet("N ＝ 品質と計算量のトレードオフを制御するハイパーパラメータ。")
para("口頭：cache の無効化間隔と同じ概念、と system の言葉で言い換えると刺さる。", italic=True)

# 13
slide(13, "評価設定", "1:00", "設定表＋『ほぼ同等』の定義")
kv("見出し", "ImageNet 384² クラス条件付き生成 / 3モデル")
bullet("モデル：H-MaskGIT-L(480M) / B(142M) / S(69M)。")
bullet("Halton, T=32, sched_pow=2, top-k=−1、CFG=0.5/0.7/1.0。")
bullet("float32, 4GPU DDP, batch=32, seed=42, 各設定 50,000枚。")
kv("指標", "品質＝FID↓・IS↑ / 効率＝理論 FLOPs 高速化率↑")
kv("重要な定義", "『ほぼ同等』＝ +0.5 FID budget 内")
para("口頭：『ほぼ同等』を先に数値定義しておく＝後の主張の逃げ道を塞ぐ。誠実さアピール。", italic=True)

# 14
slide(14, "結果①：リフレッシュ無しは破綻", "1:30", "図7上（L）、N→∞ を赤丸強調")
kv("見出し", "純粋な部分更新（リフレッシュ無し）では品質が破綻")
kv("数字強調", "H-MaskGIT-L：FID 2.541 → 5.937（N→∞）")
kv("一文", "キャッシュ再利用だけでは品質を維持できない ＝ リフレッシュが本質")
para("口頭：ここを山場に。ネガティブ結果を主結果として提示＝『無効化間隔の設計が肝』という教訓。", italic=True)

# 15
slide(15, "結果②：N トレードオフとモデル規模依存", "2:00", "表1＋図7")
kv("見出し", "N で品質・速度を制御 — 小モデルほどキャッシュ耐性が高い")
table(
    ["モデル", "設定", "FID↓", "高速化↑"],
    [
        ["S (69M)", "baseline / N=9 / N=13", "6.101 / 6.109 / 6.311", "1.00× / 1.28× / 1.30×"],
        ["B (142M)", "baseline / N=4 / N=9", "4.202 / 4.215 / 4.510", "1.00× / 1.24× / 1.30×"],
        ["L (480M)", "baseline / N=2 / N=4", "2.541 / 2.665 / 3.023", "1.00× / 1.23× / 1.37×"],
    ],
)
bullet("S：N=9 でほぼ無損失（+0.008）で 1.28×。")
bullet("L：N=2 が実用限界（+0.12）、N≥5 で劣化顕著。")
bullet("+0.5 budget 内で 18〜30% の理論削減。")
para("口頭：『約23%』ではなく動作点を明示。規模依存の考察を添えると深みが出る。", italic=True)

# 16
slide(16, "定性比較", "0:30", "図6（3列）")
kv("見出し", "リフレッシュ有りはベースラインと同等の見た目")
kv("キャプション", "図6：左=baseline、中=リフレッシュ無し（崩れ）、右=提案 N=4")
para("口頭：no-refresh 列の崩れを指差し、数値（Slide14）と視覚が一致、と締める。", italic=True)

# 17
slide(17, "まとめ・今後", "1:00", "まとめ／今後の2カラム")
kv("見出し", "まとめ")
bullet("Halton の事前決定性を利用し、追加学習なしで FFN 出力をキャッシュ再利用。")
bullet("周期的リフレッシュで L/B/S 全モデルがほぼ同等 FID で理論 FLOPs を削減。")
kv("今後", "")
bullet("注意機構への拡張（active 位置のみクエリ計算）。")
bullet("理論 FLOPs だけでなく実測レイテンシでの評価。")
bullet("リフレッシュ周期 N の適応的制御。")
para("口頭：今後の『実測レイテンシ』は自分から言う→質疑を先回り。バックアップに数字がある状態で言うと強い。", italic=True)

doc.add_page_break()

# =========================================================
# バックアップ
# =========================================================
h("バックアップスライド（本編後・質疑用、優先度順）", level=1)
bullet("B1 実測レイテンシ（最優先で作る）：transformer-only latency、理論 vs 実測、gather/scatter オーバーヘッド。速報：L N=2 実測 約1.11×（理論1.225×）。")
bullet("B2 メモリオーバーヘッド：L×(b,577,d) のキャッシュ増加量を実数値で。")
bullet("B3 図4：層・ステップ・トークン齢の詳細分解。")
bullet("B4 注意機構への予備実験：attn cache N=2 で FID 2.700。")
bullet("B5 全 N の FID/IS 生データ表（L/B/S フル sweep）。")

doc.add_paragraph()

h("想定質疑と回答方針", level=1)
table(
    ["想定質問", "回答方針"],
    [
        ["実測速度は？", "B1 バックアップ。transformer-only で理論に近づく＋乖離要因を説明。"],
        ["なぜ FFN だけ？注意機構は？", "図1で FFN 61.7%。attn は K/V を全token必要とし設計が非自明。予備で attn cache N=2→FID 2.700。"],
        ["バッチ内で active 数が揃わない場合は？", "Halton は全サンプル共通スケジュール → 揃う。Halton 採用の副次的利点。"],
        ["メモリは増えないのか", "B2 バックアップ。層数×(b,577,d) の増加を実数値で。"],
        ["他の高速化と直交するか", "直交する。ただし step 数を減らすとキャッシュの効きは落ちる、と正直に。"],
    ],
)

doc.add_paragraph()
h("生成ツールに渡す時の1行サマリ", level=1)
para("17枚＋バックアップ5枚、16:9。各スライドは『主張の一文』を見出しに。図はキャプションのみ日本語化。Slide 6 は比較表、Slide 9 は3段アニメ（トークン分類→層ゲート→層内部）、Slide 10 は数式1つ、Slide 14/15 が結果の山場。赤は U_t と警告に限定。数値は動作点を明示（『約23%』の丸めは避ける）。")

out = "/work/q-li/Halton-MaskGIT/SWoPP2026_CPSY_slide_spec.docx"
doc.save(out)
print("saved:", out)
