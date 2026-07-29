import { fileURLToPath } from "node:url";

import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, repositoryRoot, "");
  const accessToken = env.CLEARFRAME_ACCESS_TOKEN;

  return {
    envDir: repositoryRoot,
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: "http://127.0.0.1:8000",
          changeOrigin: true,
          headers: accessToken
            ? { "X-Access-Token": accessToken }
            : undefined,
        },
      },
    },
  };
});
