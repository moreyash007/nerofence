"use client";

import { useEffect, useState } from "react";

export type Theme = "dark" | "light";

/** Reads the theme the layout's bootstrap script (or the OS) already settled on. */
function currentTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  const stamped = document.documentElement.getAttribute("data-theme");
  if (stamped === "dark" || stamped === "light") return stamped;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/**
 * Theme switch. Exposes the active theme upward so the canvas can rebuild its
 * LUT — dark mode is a *selected* set of steps, not an inverted light ramp.
 */
export default function ThemeToggle({ onChange }: { onChange?: (t: Theme) => void }) {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const initial = currentTheme();
    setTheme(initial);
    onChange?.(initial);

    const media = window.matchMedia?.("(prefers-color-scheme: light)");
    if (!media) return;
    const handle = () => {
      if (document.documentElement.hasAttribute("data-theme")) return; // user chose
      const next = currentTheme();
      setTheme(next);
      onChange?.(next);
    };
    media.addEventListener("change", handle);
    return () => media.removeEventListener("change", handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("nf-theme", next);
    } catch {
      /* private mode / blocked storage — the in-memory choice still applies */
    }
    onChange?.(next);
  }

  return (
    <button
      type="button"
      className="btn"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      {theme === "dark" ? "☀ Light" : "☾ Dark"}
    </button>
  );
}
