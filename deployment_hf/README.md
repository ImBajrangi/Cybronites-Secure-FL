---
title: Secure Federated Learning System
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Secure Federated Learning System

This is a production-ready Federated Learning (FL) Laboratory and Institutional IDE.

## 🚀 Deployment Instructions

1.  **Create a New Space**: Choose **Docker** as the SDK.
2.  **Upload the Code**: Push this repository to the HF Space.
3.  **Configure Secrets**:
    - Go to **Settings > Variables and Secrets**.
    - Add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`.
4.  **Update Supabase Redirects**:
    - Copy the Space URL (e.g., `https://user-space.hf.space`).
    - Add it to your Supabase Project **Authentication > URL Configuration > Redirect URIs**.

## 🏗️ Architecture

- **Backend**: FastAPI + Flower (Federated Learning Orchestrator).
- **Frontend**: React + Tailwind (Institutional Dashboard).
- **Security**: Diffie-Hellman Key Exchange, Differential Privacy, and Blockchain Ledger.

---
*Maintained by the AI Guardian Team.*
