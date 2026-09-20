"use client";

import React, { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Search, Filter, ArrowUpDown, ArrowRight, Eye, X, Table, Box } from "lucide-react";
import { fetchCandidates } from "@/lib/api";
import { CandidateListItem } from "@/types/investigator";

const CandidateField3D = dynamic(() => import("@/components/three/CandidateField3D"), {
  ssr: false,
  loading: () => <div className="h-96 rounded bg-[#0a0e17] border border-zinc-800 animate-pulse" />
});

export default function CandidateBrowserPage() {
  const [candidates, setCandidates] = useState<CandidateListItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<"table" | "3d">("table");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedClass, setSelectedClass] = useState<string>("ALL");
  const [sortField, setSortField] = useState<"candidate_id" | "clean_observations" | "valid_tokens" | "baseline_days">("candidate_id");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [previewCandidate, setPreviewCandidate] = useState<CandidateListItem | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const res = await fetchCandidates();
        setCandidates(res.candidates || []);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const uniqueClasses = useMemo(() => {
    const s = new Set<string>();
    candidates.forEach((c) => {
      if (c.class) s.add(c.class);
    });
    return ["ALL", ...Array.from(s).sort()];
  }, [candidates]);

  const filteredCandidates = useMemo(() => {
    return candidates
      .filter((c) => {
        const matchesSearch =
          c.candidate_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
          c.ztf_designation.toLowerCase().includes(searchQuery.toLowerCase()) ||
          c.class.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesClass = selectedClass === "ALL" || c.class.toLowerCase() === selectedClass.toLowerCase();
        return matchesSearch && matchesClass;
      })
      .sort((a, b) => {
        const valA = a[sortField];
        const valB = b[sortField];
        if (typeof valA === "string") {
          return sortOrder === "asc" ? valA.localeCompare(valB as string) : (valB as string).localeCompare(valA);
        }
        return sortOrder === "asc" ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
      });
  }, [candidates, searchQuery, selectedClass, sortField, sortOrder]);

  const toggleSort = (field: "candidate_id" | "clean_observations" | "valid_tokens" | "baseline_days") => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 relative">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono text-cyan-400 font-semibold uppercase tracking-wider">
            Curated Benchmark Population
          </span>
        </div>
        <h1 className="text-2xl font-bold font-mono text-white">Candidate Registry & Catalog Browser</h1>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
          Browse 106 primary frozen real-ZTF candidates across confirmed in-distribution, out-of-distribution anomaly, and unclassified stellar archetypes.
        </p>
      </div>

      {/* Search and Filters Bar */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-4 rounded-lg bg-[#0d1322] border border-zinc-800">
        <div className="relative w-full md:w-96">
          <Search className="w-4 h-4 text-zinc-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search candidate ID, ZTF name, or class..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-zinc-900 border border-zinc-700/70 rounded-md text-xs font-mono text-white focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto overflow-x-auto">
          <div className="flex items-center gap-1.5 text-xs font-mono text-zinc-400 shrink-0">
            <Filter className="w-3.5 h-3.5 text-zinc-400" />
            <span>Class Filter:</span>
          </div>
          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="bg-zinc-900 border border-zinc-700/70 rounded-md text-xs font-mono text-white px-3 py-2 focus:outline-none focus:border-cyan-500 transition-colors"
          >
            {uniqueClasses.map((cls) => (
              <option key={cls} value={cls}>
                {cls}
              </option>
            ))}
          </select>

          {/* View Mode Toggle: TABLE vs 3D FIELD */}
          <div className="flex items-center rounded bg-zinc-900 border border-zinc-700/70 p-0.5 text-xs font-mono shrink-0">
            <button
              onClick={() => setViewMode("table")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded transition-colors ${
                viewMode === "table"
                  ? "bg-cyan-950 text-cyan-300 font-bold border border-cyan-800"
                  : "text-zinc-400 hover:text-white"
              }`}
            >
              <Table className="w-3.5 h-3.5" />
              <span>TABLE</span>
            </button>
            <button
              onClick={() => setViewMode("3d")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded transition-colors ${
                viewMode === "3d"
                  ? "bg-cyan-950 text-cyan-300 font-bold border border-cyan-800"
                  : "text-zinc-400 hover:text-white"
              }`}
            >
              <Box className="w-3.5 h-3.5" />
              <span>3D FIELD</span>
            </button>
          </div>

          <div className="text-xs font-mono text-zinc-400 px-3 py-2 bg-zinc-900/80 rounded border border-zinc-800 shrink-0">
            Showing <span className="text-white font-bold">{filteredCandidates.length}</span> candidates
          </div>
        </div>
      </div>

      {/* Candidates Display: 3D Field vs Table */}
      {viewMode === "3d" ? (
        <CandidateField3D candidates={filteredCandidates} />
      ) : (
        <div className="rounded-lg border border-zinc-800/80 bg-[#090d16] overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#0f1624] text-zinc-400 uppercase tracking-wider text-[10px] border-b border-zinc-800">
              <tr>
                <th className="py-3 px-4 cursor-pointer hover:text-white" onClick={() => toggleSort("candidate_id")}>
                  <div className="flex items-center gap-1">
                    <span>Candidate ID</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-4">ZTF Designation</th>
                <th className="py-3 px-4">Astrophysical Class</th>
                <th className="py-3 px-4">Coordinates (RA, Dec)</th>
                <th className="py-3 px-4 cursor-pointer hover:text-white" onClick={() => toggleSort("clean_observations")}>
                  <div className="flex items-center gap-1">
                    <span>Clean Obs</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-4 cursor-pointer hover:text-white" onClick={() => toggleSort("valid_tokens")}>
                  <div className="flex items-center gap-1">
                    <span>Tokens (50)</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-4 cursor-pointer hover:text-white" onClick={() => toggleSort("baseline_days")}>
                  <div className="flex items-center gap-1">
                    <span>Baseline (d)</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-4">Passbands</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
              {loading ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-zinc-500 font-mono">
                    Loading benchmark registry from NASA IPAC archive...
                  </td>
                </tr>
              ) : filteredCandidates.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-zinc-500 font-mono">
                    No matching astronomical candidates found.
                  </td>
                </tr>
              ) : (
                filteredCandidates.map((c) => (
                  <tr key={c.candidate_id} className="hover:bg-zinc-800/30 transition-colors">
                    <td className="py-3 px-4 font-bold text-white">
                      <Link href={`/investigate/${c.candidate_id}`} className="hover:text-cyan-300">
                        {c.candidate_id}
                      </Link>
                    </td>
                    <td className="py-3 px-4 text-zinc-400">{c.ztf_designation}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded text-[11px] bg-zinc-900 border border-zinc-800 text-cyan-300">
                        {c.class}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-zinc-400">
                      {c.ra.toFixed(4)}°, {c.dec.toFixed(4)}°
                    </td>
                    <td className="py-3 px-4 text-zinc-200">{c.clean_observations}</td>
                    <td className="py-3 px-4">
                      <span className="font-semibold text-emerald-400">{c.valid_tokens}</span>
                      <span className="text-zinc-600"> / 50</span>
                    </td>
                    <td className="py-3 px-4 text-zinc-300">{c.baseline_days.toFixed(1)}</td>
                    <td className="py-3 px-4">
                      <div className="flex gap-1">
                        {c.available_filters.map((b) => (
                          <span
                            key={b}
                            className={`px-1 rounded text-[10px] uppercase ${
                              b === "zg" ? "bg-emerald-950/80 text-emerald-400" : b === "zr" ? "bg-rose-950/80 text-rose-400" : "bg-amber-950/80 text-amber-400"
                            }`}
                          >
                            {b}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-right flex items-center justify-end gap-2">
                      <button
                        onClick={() => setPreviewCandidate(c)}
                        className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs transition-colors"
                        title="Quick Preview"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                      <Link
                        href={`/investigate/${c.candidate_id}`}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-cyan-950/60 hover:bg-cyan-900/80 border border-cyan-800 text-cyan-300 text-xs font-semibold transition-colors"
                      >
                        <span>Investigate</span>
                        <ArrowRight className="w-3 h-3" />
                      </Link>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
      )}

      {/* Quick Candidate Preview Drawer / Modal */}
      {previewCandidate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-lg bg-[#0a0e17] border border-cyan-500/80 shadow-2xl p-6 space-y-4 font-mono">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <div className="text-[10px] text-cyan-400 uppercase font-bold">Candidate Quick Preview</div>
                <h3 className="text-lg font-bold text-white">{previewCandidate.candidate_id}</h3>
              </div>
              <button
                onClick={() => setPreviewCandidate(null)}
                className="p-1 rounded text-zinc-400 hover:text-white hover:bg-zinc-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">ZTF Designation</span>
                <span className="text-white font-bold">{previewCandidate.ztf_designation}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Astrophysical Class</span>
                <span className="text-cyan-300 font-bold">{previewCandidate.class}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Coordinates (RA, Dec)</span>
                <span className="text-white font-bold">{previewCandidate.ra.toFixed(5)}°, {previewCandidate.dec.toFixed(5)}°</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Raw Observations</span>
                <span className="text-white font-bold">{previewCandidate.raw_observations}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Clean Filtered Observations</span>
                <span className="text-emerald-400 font-bold">{previewCandidate.clean_observations}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Valid Token Count</span>
                <span className="text-emerald-400 font-bold">{previewCandidate.valid_tokens} / 50</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Time Baseline</span>
                <span className="text-white font-bold">{previewCandidate.baseline_days.toFixed(1)} days</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-zinc-400">Available Passbands</span>
                <span className="text-amber-300 font-bold">{previewCandidate.available_filters.join(", ")}</span>
              </div>
            </div>

            <div className="pt-3 border-t border-zinc-800 flex justify-end gap-3">
              <button
                onClick={() => setPreviewCandidate(null)}
                className="px-4 py-2 rounded bg-zinc-900 border border-zinc-800 text-xs text-zinc-300 hover:text-white"
              >
                Close
              </button>
              <Link
                href={`/investigate/${previewCandidate.candidate_id}`}
                className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold flex items-center gap-1.5"
              >
                <span>Launch Workstation</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
