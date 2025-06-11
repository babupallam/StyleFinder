import React, { useState, useEffect } from "react";
import UploadBox from "../components/UploadBox";
import ImageGallery from "../components/ImageGallery";
import axios from "axios";
import TopKSelector from "../components/TopKSelector";
import ModelDescription from "../components/ModelDescription";

type SearchResult = {
  url: string;
  score: number;
  rank: number;
  label?: string;
};

const SearchPage: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);

  const [modelConfigs, setModelConfigs] = useState<any>({});
  const [availableArchitectures, setAvailableArchitectures] = useState<string[]>([]);
  const [selectedArch, setSelectedArch] = useState<string>("");
  const [modelNames, setModelNames] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelDescription, setModelDescription] = useState<{ [key: string]: string }>({});


  useEffect(() => {
    axios.get("http://localhost:8000/api/models")
      .then(res => {
        const configs = res.data;
        setModelConfigs(configs);
        const archs = Object.keys(configs);
        setAvailableArchitectures(archs);
        if (archs.length > 0) {
          //setSelectedArch(archs[0]);
          const modelKeys = Object.keys(configs[archs[0]]);
          setModelNames(modelKeys);
          //setSelectedModel(modelKeys[0]);
        }
      })
      .catch(err => console.error("Error loading model configs", err));
  }, []);



  const [topK, setTopK] = useState<number>(10);

  const [loading, setLoading] = useState(false);

  const handleUpload = (file: File) => {
    setImageFile(file);
    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  const handleArchChange = (arch: string) => {
    setSelectedArch(arch);
    const models = Object.keys(modelConfigs[arch] || {});
    setModelNames(models);
    setSelectedModel(models[0]);
  };

  const handleModelChange = async (selectedModel: string) => {
    setSelectedModel(selectedModel);

    try {
      const encodedArch = encodeURIComponent(selectedArch);
      const res = await fetch(`http://localhost:8000/api/model-description/${encodeURIComponent(selectedArch)}/${selectedModel}`);
      const data = await res.json();
      if (data.description) setModelDescription(data.description);
      else setModelDescription({});
    } catch (err) {
      console.error("Failed to fetch model description", err);
      setModelDescription({});
    }
  };

  useEffect(() => {
    const handlePaste = (e: Event) => {
      const event = e as ClipboardEvent; //  type assertion
      const items = event.clipboardData?.items;
      if (items) {
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) {
              setImageFile(file);
              const reader = new FileReader();
              reader.onloadend = () => setImagePreview(reader.result as string);
              reader.readAsDataURL(file);
            }
          }
        }
      }

    };

    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);


  const handleSearch = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("use_finetuned", "false");
    formData.append("top_k", topK.toString());
    formData.append("clip_arch", selectedArch);
    formData.append("model", selectedModel);  // this is like "baseline", "stage3_joint_053013", etc.

   

    try {
      setLoading(true);
      console.log(" Sending image to backend...");

      const res = await axios.post("http://localhost:8000/api/search", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log(" Response from backend:", res.data);
      setResults(res.data.similar_images); //  this updates the gallery
    } catch (err: any) {
      console.error(" Search failed", err);
      alert(err.response?.data?.detail || "An error occurred while searching.");
    }
    finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gray-50 px-1 py-2 flex flex-col items-center">
      <div className="w-full max-w-10xl mx-auto grid sm:grid-cols-3 gap-10">

        {/* Left 2/3rds: Search Block */}
        <div className="col-span-2 bg-white p-5 rounded-2xl shadow-lg">
          <h1 className="text-3xl font-bold mb-6 flex items-center gap-1 text-gray-800">
            <span>🔍</span> Visual Style Search
          </h1>

          {/* Visual Upload + Dropdown Section (same as your existing layout) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-start">

            {/* Upload Box */}
            <div className="flex flex-col items-center justify-center">
              <div className="h-60 border-2 border-dashed border-blue-900 rounded-xl p-4 flex flex-col items-center justify-between bg-white shadow-sm hover:shadow-md transition-all">
                <UploadBox onImageUpload={handleUpload} />
                <div className="w-full h-24 mt-2 rounded-md overflow-hidden bg-gray-50 border">
                  <img
                    src={imagePreview || "https://via.placeholder.com/100x120?text=No+Image"}
                    alt="preview"
                    className="object-contain w-full h-full"
                  />
                </div>
                <p className="text-xs text-gray-400 text-center mt-1">Click or drag to upload</p>
              </div>
              <p className="text-xs text-gray-400 mt-2 text-center">
                You can also paste a screenshot (Ctrl+V) or drag an image here.
              </p>
            </div>


            {/* Form Controls */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 block">Select Architecture</label>
              <select
                value={selectedArch}
                onChange={(e) => handleArchChange(e.target.value)}
                className="w-full px-4 py-2 mb-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
              >
                <option value="">-- Select Architecture --</option>
                {availableArchitectures.map((arch) => (
                  <option key={arch} value={arch}>
                    {arch}
                  </option>
                ))}
              </select>


              <div className="mt-3 space-y-3">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">Select Model</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => handleModelChange(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                  >
                    <option value="">-- Select Model --</option>
                    {modelNames.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>

                </div>

                <TopKSelector topK={topK} setTopK={setTopK} />

                <button
                  onClick={handleSearch}
                  disabled={loading || !imageFile}
                  className={`w-full py-3 rounded-md text-white font-semibold shadow transition-all ${loading || !imageFile ? "bg-gray-300 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
                    }`}
                >
                  {loading ? "Searching..." : "Search"}
                </button>
              </div>
            </div>
          </div>

          {/* Loader / Results */}
          {loading ? (
            <div className="mt-10 text-center">
              <div className="w-12 h-12 border-4 border-blue-300 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-500">Searching for similar images...</p>
            </div>
          ) : results.length > 0 && (
            <div className="mt-10">
              <h2 className="text-xl font-semibold text-gray-800 mb-4"> Similar Images</h2>
              <ImageGallery results={results} />
            </div>
          )}
        </div>

        {/* Right Column: Model Configuration */}
        <div className="bg-white rounded-2xl shadow-md p-5 h-fit">
          <h3 className="font-semibold text-gray-700 mb-3 text-lg flex items-center gap-2">
             Model Configuration
          </h3>
          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
            {Object.entries(modelDescription).map(([key, val]) => (
              <li key={key}><strong>{key}:</strong> {val}</li>
            ))}
            {Object.keys(modelDescription).length === 0 && (
              <li className="text-gray-400 italic">No details available</li>
            )}
          </ul>
        </div>
      </div>
      <div className="fixed bottom-4 right-4 bg-yellow-100 text-yellow-900 text-xs px-4 py-2 rounded-lg shadow-md z-50">
         Development Demo – Features may change
      </div>

    </div >

  );
};

export default SearchPage;
