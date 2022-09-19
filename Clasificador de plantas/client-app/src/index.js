import React from "react";
import { createRoot } from 'react-dom/client';

// Bootstrap CSS
import "bootstrap/dist/css/bootstrap.min.css";
// Bootstrap Bundle JS
import "bootstrap/dist/js/bootstrap.bundle.min";
import "./index.css";
import App from "./App";
const container = document.getElementById("root");
const root = createRoot(container);
root.render(<App />);
