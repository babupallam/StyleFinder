import React from "react";

type Props = {
  description: { [key: string]: string };
};

const ModelDescription: React.FC<Props> = ({ description }) => {
  return (
    <div className="mt-4 p-4 border rounded bg-gray-50">
      <h3 className="text-md font-semibold mb-2 text-gray-700"> Model Configuration</h3>
      <ul className="text-sm text-gray-600 list-disc pl-5">
        {Object.entries(description).map(([key, value]) => (
          <li key={key}>
            <span className="font-medium">{key}:</span> {value}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ModelDescription;
