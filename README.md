# Trading Website Demo

This repo contains a small demo app with a React frontend, a Node/Express backend and a native addon built with `node-gyp`.

## Prerequisites

- Node.js 18+
- npm
- A running MongoDB instance (local or remote)

## Installation

1. Install dependencies for each part:

```bash
cd backend && npm install
cd ../frontend && npm install
cd ../addon && npm install  # optional: builds the native addon
```

2. Copy `backend/.env.example` to `backend/.env` and adjust values if necessary.

3. Start MongoDB locally (`docker run --rm -p 27017:27017 mongo`) or point `MONGO_URI` to another instance.

## Running

In separate terminals run:

```bash
cd backend && npm run dev
```

```bash
cd frontend && npm run dev
```

The frontend dev server proxies API requests to the backend on port `8000` by default.

## Notes

The `cpp_engine` folder contains minimal stub implementations so the optional native addon compiles. It does not perform any trading logic.

## Frontend verification

```sh
cd frontend
npm ci
npm run build
npm run dev
```

The production build performs TypeScript checks before bundling. A successful
frontend build verifies compilation only; backtests still require the backend,
its database, datasets, and native addon described above. Research/code-generation
and collaboration screens currently include sample content or unavailable actions;
they should not be presented as live AI or multi-user workflows in a demo.

Browser checks now exercise all six routes, accessible navigation, dataset
selection, and actual unavailable-backend error rendering at 1440px and 390px.
Install frontend dependencies and Chromium using `npx playwright-core install
chromium`, then run `node scripts/check_ui_browser.cjs`. `CHROME_BIN` can select
a local Chromium binary. This gate verifies UI/navigation/error behavior; it
does not establish successful backtests, authentication, or live portfolio data.
