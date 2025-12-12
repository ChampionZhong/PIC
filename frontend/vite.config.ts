import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8001,
    host: 'localhost',
    watch: {
      // 使用轮询模式替代文件监视器，避免 inotify 限制
      usePolling: true,
      interval: 1000,
      // 排除 node_modules 和其他不需要监视的目录
      ignored: ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/.next/**', '**/build/**']
    }
  },
  // 优化构建性能
  optimizeDeps: {
    exclude: []
  }
})

