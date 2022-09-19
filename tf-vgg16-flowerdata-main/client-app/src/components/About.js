import React from "react";
import "../Tabs.css";
import Participants from "./Participants";

function About() {
  return (
    <div className="about-container">
      <h1>Sobre el proyecto</h1>
      <p>
        Este proyecto forma parte de un trabajo del curso CI-0163 Análisis de
        Grandes Volúmenes de Datos de la carrera de Bachillerato en computación
        con varios énfasis. Tiene como objetivo el desarrollar un prototipo de
        aplicación web para facilitar el uso de modelos de aprendizaje máquina
        para clasificación de plantas.
      </p>
      {/* <p>
        Si te interesa aprender más sobre cómo funciona puedes ver el
        repositorio{" "}
        <a href="https://github.com/Solrak97/tf-vgg16-flowerdata">aquí</a>
      </p> */}
      <h1>Participantes</h1>
      <Participants />
    </div>
  );
}

export default About;
