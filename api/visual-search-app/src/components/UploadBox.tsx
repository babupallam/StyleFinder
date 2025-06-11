import React from "react";

type UploadBoxProps = {
  onImageUpload: (file: File) => void;
};

const UploadBox: React.FC<UploadBoxProps> = ({ onImageUpload }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) onImageUpload(e.target.files[0]);
  };

  return (
    <label className="block border border-gray-300 rounded-md p-5 bg-gray-50 hover:border-blue-500 transition cursor-pointer text-center">
      <input type="file" accept="image/*" className="hidden" onChange={handleChange} />
      <div className="text-blue-600 font-medium"> Click to upload image</div>
    </label>
  );
};

export default UploadBox;
