import { useEffect } from "react";

// Reading focus for a page built from revealed steps.
//
// Whatever sits under the middle of the viewport is shown at full strength, and
// the sections above and below fade out in proportion to how far away they are.
// It follows the scroll in both directions, so a participant who scrolls back up
// to re-read an earlier step sees that step return to full colour and the rest
// recede. This is what makes the auto-scroll after each choice feel like the page
// is handing them the next thing to do, rather than just jumping.
//
// Purely cosmetic: nothing is hidden or disabled, the faded text is still
// readable, and hovering or tabbing into a faded section restores it at once
// (see .focusblock in styles.css).

// Never fade so far that the text stops being legible for someone who wants to
// read ahead. 0.45 keeps body copy comfortably readable on both themes.
const MIN_OPACITY = 1;
// How far a section has to sit from the middle of the viewport before it reaches
// MIN_OPACITY, as a fraction of the viewport height. Smaller = sharper falloff.
const FALLOFF = 1;

// containerRef: the element holding the sections.
// Every descendant with class "focusblock" is treated as one section, so the
// blocks must not be nested inside one another (opacity would compound).
// deps: re-run when sections are added or removed, so newly revealed steps are
// graded immediately rather than at the next scroll event.
export function useReadingFocus(containerRef, deps = []) {
  useEffect(() => {
    const root = containerRef.current;
    if (!root) return;
    // Scroll-linked fading can be uncomfortable for people who ask for reduced
    // motion, so they simply get the whole page at full strength.
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (calm.matches) return;

    let frame = 0;

    const paint = () => {
      frame = 0;
      const mid = window.innerHeight / 2;
      root.querySelectorAll(".focusblock").forEach((el) => {
        const box = el.getBoundingClientRect();
        // Distance from the middle of the viewport to the nearest edge of the
        // section, so a section that spans the middle counts as fully in focus.
        // (Measuring to its centre instead would fade tall sections, like the
        // fit gate with its objective panel, while they are being read.)
        const gap =
          box.top > mid
            ? box.top - mid
            : box.bottom < mid
              ? mid - box.bottom
              : 0;
        const t = Math.min(1, gap / (window.innerHeight * FALLOFF));
        el.style.opacity = String(1 - t * (1 - MIN_OPACITY));
      });
    };

    // One update per animation frame at most: the handler fires continuously
    // during a smooth scroll, and reading layout on every event would thrash.
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(paint);
    };

    paint();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (frame) cancelAnimationFrame(frame);
      // Hand the sections back to the stylesheet, so a section that unmounts
      // and returns (for example via Back) is never stuck faded.
      root.querySelectorAll(".focusblock").forEach((el) => {
        el.style.opacity = "";
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
