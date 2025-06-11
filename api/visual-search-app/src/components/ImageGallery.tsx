import React, { useState } from "react";

type SearchResult = {
  url: string;
  score: number;
  rank: number;
  label?: string;
};

const ITEMS_PER_PAGE = 10;

const ImageGallery: React.FC<{ results: SearchResult[] }> = ({ results }) => {
  const [currentPage, setCurrentPage] = useState(0);

  const startIndex = currentPage * ITEMS_PER_PAGE;
  const paginatedResults = results.slice(startIndex, startIndex + ITEMS_PER_PAGE);
  const totalPages = Math.ceil(results.length / ITEMS_PER_PAGE);

  return (
    <div className="flex flex-col gap-8">
      {/* Image Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
        {paginatedResults.map((result) => (
          <div
            key={result.rank}
            className="bg-white p-3 rounded-xl shadow hover:shadow-xl transition-all"
          >
            <img
              src={`http://localhost:8000${result.url}`}
              alt={result.label}
              className="w-full h-44 object-cover rounded-md mb-2"
            />
            <p className="text-sm text-gray-500">Rank #{result.rank}</p>
            <p className="text-base font-medium text-gray-700">{result.label}</p>
            <p className="text-xs text-blue-600">
              Visual Similarity Score: {(result.score).toFixed(2)}%
            </p>
          </div>
        ))}
      </div>

      {/* Pagination Buttons */}
      {totalPages > 1 && (
        <div className="flex justify-center gap-4">
          <button
            onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 0))}
            className={`px-4 py-2 rounded-lg font-semibold ${
              currentPage === 0
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-blue-600 text-white hover:bg-blue-700"
            }`}
            disabled={currentPage === 0}
          >
            ← Previous
          </button>
          <button
            onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages - 1))}
            className={`px-4 py-2 rounded-lg font-semibold ${
              currentPage >= totalPages - 1
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-blue-600 text-white hover:bg-blue-700"
            }`}
            disabled={currentPage >= totalPages - 1}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageGallery;
