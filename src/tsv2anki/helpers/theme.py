import hashlib

FRONT_TEMPLATE = """
<div class="card" style="background-color: {{CategoryColor}}">
  <div class="header" style="color: {{SubCategoryColor}};">{{SubCategory}}</div>
  <div class="front">{{Front}}</div>
  <div class="example">{{Example}}</div>
</div>
"""

BACK_TEMPLATE = """
<div class="card" style="background-color: {{CategoryColor}}">
  <div class="header" style="color: {{SubCategoryColor}};">{{SubCategory}}</div>
  <div class="front">{{Front}}</div>
  <div class="example">{{Example}}</div>
  <hr>
  <div class="back">{{Back}}</div>
  <div class="example">{{TranslatedExample}}</div>
</div>
"""

FLIPPED_FRONT_TEMPLATE = """
<div class="card" style="background-color: {{CategoryColor}}">
  <div class="header" style="color: {{SubCategoryColor}};">{{SubCategory}}</div>
  <div class="front">{{Back}}</div>
  <div class="example">{{TranslatedExample}}</div>
</div>
"""

FLIPPED_BACK_TEMPLATE = """
<div class="card" style="background-color: {{CategoryColor}}">
  <div class="header" style="color: {{SubCategoryColor}};">{{SubCategory}}</div>
  <div class="front">{{Back}}</div>
  <div class="example">{{TranslatedExample}}</div>
  <hr>
  <div class="back">{{Front}}</div>
  <div class="example">{{Example}}</div>
</div>
"""

CARD_CSS = """
.card {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  text-align: center;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.15);
  padding: 20px;
  margin: 10px auto;
  width: 90%;
  max-width: 500px;
  transition: background 0.3s ease, color 0.3s ease;
}

/* Header section */
.header {
  font-size: 1.2em;
  font-weight: bold;
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* Front and back text */
.front {
  font-size: 1.4em;
  margin-bottom: 10px;
}

.back {
  font-size: 1.2em;
  color: #333;
}

.example {
  font-size: 0.8em;
  color: #333;
}


/* Optional hover effect (for preview mode) */
.card:hover {
  transform: scale(1.02);
  box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
"""


def color_from_string(s: str, is_background: bool=True) -> str:
    s =s.lower().strip()
    h = hashlib.md5(s.encode()).hexdigest()
    hue = int(h[:2], 16) * 360 / 255
    if is_background:
        return f"hsl({hue}, 70%, 85%)"
    else:
        return f"hsl({hue}, 70%, 20%)"
