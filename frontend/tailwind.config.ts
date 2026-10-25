import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        lavender: "#6E56CF",
        "lavender-dark": "#3A2E6E",
        ink: "#1A1523",
      },
    },
  },
  plugins: [],
};
export default config;
