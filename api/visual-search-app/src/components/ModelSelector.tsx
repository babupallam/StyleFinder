import React from "react";

type Props = {
  selectedModel: string;
  onChange: (model: string) => void;
};

const models = ["baseline", "finetuned_v1", "finetuned_v2"];

const ModelSelector: React.FC<Props> = ({ selectedModel, onChange }) => {
  return (
    <select
      value={selectedModel}
      onChange={(e) => onChange(e.target.value)}
      className="px-4 py-2 rounded-md border border-gray-300 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
    >
      {models.map((model) => (
        <option key={model} value={model}>
          {model}
        </option>
      ))}
    </select>
  );
};

export default ModelSelector;
