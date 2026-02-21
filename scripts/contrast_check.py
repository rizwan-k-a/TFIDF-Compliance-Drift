from __future__ import annotations

# Simple contrast checker for key color pairs used in the UI.
# Usage: python scripts/contrast_check.py

def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(2 * c for c in h)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def _luminance(r: float, g: float, b: float) -> float:
    def srgb(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r_l = srgb(r)
    g_l = srgb(g)
    b_l = srgb(b)
    return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l


def contrast_ratio(hex1: str, hex2: str) -> float:
    r1, g1, b1 = _hex_to_rgb(hex1)
    r2, g2, b2 = _hex_to_rgb(hex2)
    L1 = _luminance(r1, g1, b1)
    L2 = _luminance(r2, g2, b2)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)


if __name__ == '__main__':
    pairs = [
        ('#6b6b63', '#FFF2E2', 'Header subtitle on page bg'),
        ('#4F633D', '#FFF2E2', 'Primary dark on page bg'),
        ('#555555', '#f7f7f7', 'Tab text on tab bg'),
        ('#4F633D', '#ffffff', 'Primary on white'),
    ]
    print('\nContrast check results:')
    for a, b, name in pairs:
        ratio = contrast_ratio(a, b)
        print(f" - {name}: {a} on {b} → {ratio:.2f}:1")
    print('\nGuidance: WCAG recommends >=4.5:1 for normal text, >=3:1 for large text.')
