"use client";
import { useEffect, useState } from "react";

export default function ThemePicker() {
  const [dark, setDark] = useState(true);
  useEffect(() => {
    const saved = localStorage.getItem("rena-theme");
    const isDark = saved !== "light";
    setDark(isDark);
    document.documentElement.dataset.renaTheme = isDark ? "dark" : "light";
  }, []);
  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.dataset.renaTheme = next ? "dark" : "light";
    localStorage.setItem("rena-theme", next ? "dark" : "light");
  }
  return <button type="button" className="theme-toggle" onClick={toggle} aria-label={`Switch to ${dark ? "light" : "dark"} theme`} title={`Switch to ${dark ? "light" : "dark"} theme`}><span>{dark ? "☼" : "☾"}</span><b>{dark ? "Dark" : "Light"}</b></button>;
}
