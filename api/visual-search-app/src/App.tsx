import React from "react";
import SearchPage from "./pages/SearchPage";
import Layout from "./components/Layout";

const App: React.FC = () => {
  return (
    <Layout>
      <SearchPage />
    </Layout>
  );
};

export default App;
