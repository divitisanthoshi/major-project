"""
UI Components - Rounded panels, progress bars, labels.
"""

import pygame
from src.ui.theme import *


def draw_rounded_rect(surface, rect, color, radius=12, border=0, border_color=None):
    """Draw a rectangle with rounded corners."""
    x, y, w, h = rect
    if radius <= 0:
        pygame.draw.rect(surface, color, rect, border)
        if border and border_color:
            pygame.draw.rect(surface, border_color, rect, border)
        return

    # Clamp radius
    radius = min(radius, w // 2, h // 2)

    # Four corner circles
    for corner_x, corner_y in [(x + radius, y + radius), (x + w - radius, y + radius),
                               (x + w - radius, y + h - radius), (x + radius, y + h - radius)]:
        pygame.draw.circle(surface, color, (corner_x, corner_y), radius)

    # Four rectangles
    pygame.draw.rect(surface, color, (x + radius, y, w - 2 * radius, h))
    pygame.draw.rect(surface, color, (x, y + radius, w, h - 2 * radius))

    if border and border_color:
        for corner_x, corner_y in [(x + radius, y + radius), (x + w - radius, y + radius),
                                   (x + w - radius, y + h - radius), (x + radius, y + h - radius)]:
            pygame.draw.circle(surface, border_color, (corner_x, corner_y), radius, border)
        pygame.draw.line(surface, border_color, (x + radius, y), (x + w - radius, y), border)
        pygame.draw.line(surface, border_color, (x + radius, y + h), (x + w - radius, y + h), border)
        pygame.draw.line(surface, border_color, (x, y + radius), (x, y + h - radius), border)
        pygame.draw.line(surface, border_color, (x + w, y + radius), (x + w, y + h - radius), border)


def draw_progress_bar(surface, rect, value, bg_color=BAR_BG, fill_colors=None):
    """Draw a styled progress bar with optional gradient by value."""
    x, y, w, h = rect
    if fill_colors is None:
        fill_colors = {0: BAR_POOR, 0.4: BAR_MODERATE, 0.7: BAR_GOOD}

    # Background
    draw_rounded_rect(surface, rect, bg_color, radius=h // 2)

    # Fill
    value = max(0, min(1, value))
    fill_w = max(0, int((w - 4) * value))
    if fill_w > 0:
        if value >= 0.7:
            fill_color = BAR_GOOD
        elif value >= 0.4:
            fill_color = BAR_MODERATE
        else:
            fill_color = BAR_POOR
        inner_h = h - 4
        inner = (x + 2, y + 2, fill_w, inner_h)
        pygame.draw.rect(surface, fill_color, inner, border_radius=inner_h // 2)


def lerp(a, b, t):
    """Smooth interpolation."""
    return a + (b - a) * min(1, max(0, t))

