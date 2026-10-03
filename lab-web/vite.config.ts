import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

const backendUrl = process.env.ZEBRAFISH_LAB_BACKEND ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: backendUrl, changeOrigin: true },
      "/ws": { target: backendUrl.replace(/^http/, "ws"), ws: true, changeOrigin: true },
    },
  },
});
