import React from "react";
import "../App.css";

function Banner() {
  return (
    <div className="background">
      <h1 className="main-title rainbow rainbow_text_animated">
        ¡Bienvenido a PlantBank!
      </h1>
      <h3 className="slogan rainbow rainbow_text_animated">
        El sitio web que te permite clasificar tipos de plantas
      </h3>
    </div>
  );
}

export default Banner;
