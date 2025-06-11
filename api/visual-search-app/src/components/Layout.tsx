import React from "react";
import { Link } from "react-router-dom";

type Props = {
  children: React.ReactNode;
};

const Layout: React.FC<Props> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/*  Top Navbar */}
      <header className="bg-white shadow-sm py-4 px-6 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-blue-600"><Link to="/" className="hover:text-blue-500" > StyleFinder</Link></h1>
        <nav className="space-x-4 text-gray-600 text-sm">
            <Link to="/" className="hover:text-blue-500" >Home</Link>
            <Link to="/shop" className="hover:text-blue-500">Shop</Link>
            <Link to="/about" className="hover:text-blue-500">About</Link>
            <Link to="/contact" className="hover:text-blue-500">Contact</Link>
        </nav>

      </header>

      {/*  Demo Banner */}
      <div className="bg-blue-100 text-blue-800 text-center py-3 text-sm font-medium">
      </div>

      {/*  Main Content */}
      <main className="flex-grow">{children}</main>

      {/* ️ Footer */}
      <footer className="bg-white text-center py-6 text-sm text-gray-400 mt-10">
        © 2025 StyleFinder. For demo purposes only.
      </footer>
    </div>
  );
};

export default Layout;
