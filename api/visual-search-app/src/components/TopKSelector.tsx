import React from "react";

type TopKSelectorProps = {
  topK: number;
  setTopK: (value: number) => void;
};

const options = [5, 10, 20, 30, 50];

const TopKSelector: React.FC<TopKSelectorProps> = ({ topK, setTopK }) => {
  return (
    <div className="flex flex-col">
      <label htmlFor="topK" className="text-sm font-medium text-gray-600 mb-1">
        Top K Results:
      </label>
      <select
        id="topK"
        value={topK}
        onChange={(e) => setTopK(parseInt(e.target.value))}
        className="px-3 py-2 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm bg-white"
      >
        {options.map((k) => (
          <option key={k} value={k}>
            Top {k}
          </option>
        ))}
      </select>
    </div>
  );
};

export default TopKSelector;
