# aws-advisor-ui

Next.js frontend for InfraMind. Provides a chat interface for describing your workload, tracks agent progress in real time, and displays the full architecture recommendation with Terraform file downloads.

> This UI talks to the FastAPI backend in the parent directory. Start the backend first before running the frontend.

---

## Prerequisites

- Node.js 18+
- npm 9+
- The backend running at `http://localhost:8000` (see the [root README](../README.md))

---

## Setup

```bash
# Install dependencies
npm install

# Copy env template
cp .env.example .env.local
```

`.env.local` only needs one line:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Running locally

```bash
npm run dev
```

Open `http://localhost:3000`.

---

## Other commands

| Command | Description |
|---------|-------------|
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | Run ESLint |

---

## Deploying to Vercel

1. Push the repo to GitHub (already done).
2. Import the repo in [Vercel](https://vercel.com/new).
3. Set the **root directory** to `aws-advisor-ui`.
4. Add an environment variable in Vercel project settings:
   ```
   NEXT_PUBLIC_API_URL=https://your-deployed-backend-url
   ```
5. Deploy.

---

## Project structure

```
aws-advisor-ui/
├── src/
│   ├── app/
│   │   ├── page.tsx           # Main chat page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/
│   │   ├── ArchSummary.tsx        # Architecture summary card
│   │   ├── AwaitingInputPanel.tsx # Follow-up question panel
│   │   ├── ConfidenceDot.tsx      # Confidence indicator
│   │   ├── ErrorPanel.tsx         # Error display
│   │   ├── ResultReport.tsx       # Full recommendation report
│   │   ├── StageProgress.tsx      # Agent progress tracker
│   │   └── TerraformViewer.tsx    # Terraform file viewer + download
│   ├── hooks/
│   │   └── useJobPoller.ts    # Polls /api/recommend/{job_id} for updates
│   └── lib/
│       ├── api.ts             # API client functions
│       └── types.ts           # TypeScript types matching backend schemas
├── .env.example
├── next.config.ts
├── tailwind.config.ts
└── package.json
```
