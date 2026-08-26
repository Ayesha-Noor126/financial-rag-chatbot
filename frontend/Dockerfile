# ==============================================================================
# Phase 3 — Production Multi-Stage Frontend Dockerfile (Vite + React + Nginx)
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: BUILD STAGE (Node.js Environment)
# ------------------------------------------------------------------------------
FROM node:20-alpine AS builder

# Set working directory inside container
WORKDIR /app

# Copy package manifest files first for Docker layer caching
COPY package.json package-lock.json ./

# Install exact npm dependencies cleanly (faster and reproducible)
RUN npm ci

# Copy application source files
COPY . .

# Accept build-time environment variables for Vite
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

# Run Vite build script to compile React app into static minified assets (/app/dist)
RUN npm run build


# ------------------------------------------------------------------------------
# STAGE 2: PRODUCTION STAGE (High-Performance Nginx Web Server)
# ------------------------------------------------------------------------------
FROM nginx:1.25-alpine

# Install curl for container health check
RUN apk add --no-cache curl

# Copy custom Nginx configuration for React SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static HTML/JS/CSS assets from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Document that Nginx serves HTTP traffic on port 80
EXPOSE 80

# Health check to ensure Nginx is actively serving the application
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start Nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"]
