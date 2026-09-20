"use client";

import React, { useEffect, useState } from "react";
import { Search } from "lucide-react";
import { searchKnowledge, fetchKnowledgeDocuments } from "@/lib/api";

export default function KnowledgeBasePage() {
  const [query, setQuery] = useState<string>("supernova light curve rise time");
  const [results, setResults] = useState<any[]>([]);
  const [documents, setDocuments] = useState<any[]>([]);
  const [totalChunks, setTotalChunks] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    async function loadDocs() {
      try {
        const docRes = await fetchKnowledgeDocuments();
        setDocuments(docRes.documents || []);
        setTotalChunks(docRes.total_chunks || 0);

        // Run default search
        const sRes = await searchKnowledge("supernova light curve rise time");
        setResults(sRes.results || []);
      } catch (err) {
        console.error(err);
      }
    }
    loadDocs();
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    try {
      setLoading(true);
      const res = await searchKnowledge(query);
      setResults(res.results || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono text-cyan-400 font-semibold uppercase tracking-wider">
            Domain Grounding & RAG Corpus
          </span>
        </div>
        <h1 className="text-2xl font-bold font-mono text-white">Astrophysical Knowledge Base</h1>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
          Search the in-memory vector store containing peer-reviewed astrophysics papers and canonical transient taxonomic catalogs.
        </p>
      </div>

      {/* RAG Telemetry and Document Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Documents Loaded</div>
          <div className="text-xl font-bold font-mono text-white mt-1">{documents.length}</div>
          <div className="text-[10px] text-cyan-400 mt-1 font-mono">Papers & Catalogs</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Indexed Chunks</div>
          <div className="text-xl font-bold font-mono text-white mt-1">{totalChunks}</div>
          <div className="text-[10px] text-zinc-400 mt-1 font-mono">Vector Embeddings</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Embedding Type</div>
          <div className="text-sm font-bold font-mono text-emerald-400 mt-1">TF-IDF Normalized</div>
          <div className="text-[10px] text-zinc-400 mt-1 font-mono">Cosine Similarity Search</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Corpus Storage</div>
          <div className="text-sm font-bold font-mono text-zinc-200 mt-1">In-Memory RAM</div>
          <div className="text-[10px] text-zinc-400 mt-1 font-mono">Zero External Network Call</div>
        </div>
      </div>

      {/* Search Input */}
      <form onSubmit={handleSearch} className="flex gap-2 p-3 rounded-lg bg-[#0a0e17] border border-zinc-800">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-zinc-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search transient taxonomy, magnetar mechanisms, shock cooling..."
            className="w-full pl-9 pr-4 py-2 bg-zinc-900 border border-zinc-700/70 rounded-md text-xs font-mono text-white focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-md text-xs font-mono font-semibold transition-colors disabled:opacity-50"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {/* Results and Corpus Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Search Results */}
        <div className="lg:col-span-2 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-zinc-300 uppercase">
              Semantic Search Results ({results.length})
            </span>
            <span className="text-[11px] font-mono text-zinc-500">Sorted by Cosine Similarity</span>
          </div>

          <div className="space-y-3">
            {results.map((res, idx) => (
              <div key={idx} className="p-4 rounded-lg bg-[#0a0e17] border border-zinc-800 space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-cyan-400 font-bold">[{res.doc_id}]</span>
                  <span className="text-emerald-400 font-semibold">Sim: {res.score.toFixed(4)}</span>
                </div>
                <h3 className="text-sm font-mono font-bold text-white">{res.title}</h3>
                <p className="text-xs text-zinc-300 leading-relaxed font-sans border-l-2 border-cyan-800 pl-3 py-1">
                  "{res.text}"
                </p>
                <div className="text-[10px] font-mono text-zinc-500 pt-1">
                  File: <code>{res.filepath}</code>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Loaded Documents */}
        <div className="space-y-3">
          <span className="text-xs font-mono font-bold text-zinc-300 uppercase">
            Loaded Corpus Documents
          </span>

          <div className="space-y-2">
            {documents.map((doc) => (
              <div key={doc.doc_id} className="p-3 rounded-lg bg-[#0d1322] border border-zinc-800 space-y-1">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-white font-bold truncate">{doc.doc_id}</span>
                  <span className="text-cyan-400 text-[10px]">{doc.chunks_count} chunks</span>
                </div>
                <div className="text-xs text-zinc-400 line-clamp-1">{doc.title}</div>
                <div className="text-[10px] font-mono text-zinc-600 truncate">
                  Type: {doc.source_type}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
