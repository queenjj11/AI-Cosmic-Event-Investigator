import {
  CandidateListItem,
  InvestigationResult,
  LightCurvePoint,
  SystemHealth
} from "@/types/investigator";

// Base API URL (proxied in next.config.mjs or direct)
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export async function fetchSystemHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch system health");
  return res.json();
}

export async function fetchCandidates(classification?: string): Promise<{ total_count: number; candidates: CandidateListItem[] }> {
  const query = classification ? `?classification=${encodeURIComponent(classification)}` : "";
  const res = await fetch(`${API_BASE}/api/candidates${query}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch candidates");
  return res.json();
}

export async function fetchInvestigation(candidateId: string): Promise<InvestigationResult> {
  const res = await fetch(`${API_BASE}/api/investigations/${encodeURIComponent(candidateId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch investigation for ${candidateId}`);
  return res.json();
}

export async function fetchLightCurve(candidateId: string): Promise<{ observations_count: number; points: LightCurvePoint[] }> {
  const res = await fetch(`${API_BASE}/api/lightcurve/${encodeURIComponent(candidateId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch light curve for ${candidateId}`);
  return res.json();
}

export async function searchKnowledge(query: string): Promise<{ count: number; results: any[] }> {
  const res = await fetch(`${API_BASE}/api/knowledge/search?q=${encodeURIComponent(query)}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to search knowledge base");
  return res.json();
}

export async function fetchKnowledgeDocuments(): Promise<{ total_documents: number; total_chunks: number; documents: any[] }> {
  const res = await fetch(`${API_BASE}/api/knowledge/documents`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch knowledge documents");
  return res.json();
}

export async function fetchReports(): Promise<{ count: number; reports: any[] }> {
  const res = await fetch(`${API_BASE}/api/reports`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch reports");
  return res.json();
}

export async function fetchMarkdownReport(candidateId: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/reports/${encodeURIComponent(candidateId)}/markdown`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch markdown report for ${candidateId}`);
  return res.text();
}
