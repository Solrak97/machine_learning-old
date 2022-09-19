import React, { useState } from "react";
import GaugeChart from 'react-gauge-chart'
import Button from "react-bootstrap/Button";
import "../App.css";
import DropZone from "./DropZone";
import Instructions from "./Instructions";

function Home() {

  const [result, setResult] = useState({ error: 'Error' });
  const [displayedResult, setDisplayResult] = useState(false);
  const [imageUploaded, setImageUploaded] = useState(undefined);


  const imageURL = imageUploaded ? URL.createObjectURL(imageUploaded) : "";

  return (
    <div>
      {displayedResult
        ?
        <div>
          {result.error
            ?
            <div>
              <h1 className="mb-5">Error al clasificar imagen</h1>
              <Button
                variant="primary"
                style={{ backgroundColor: "#31525B", fontSize: "2rem" }}
                onClick={() => { setDisplayResult(false); setImageUploaded(false); setResult({}); }}
              >
                Intentar de nuevo
              </Button>
            </div>
            :
            <div>
              <h3 className="">Clase de la planta</h3>

              <h1 className="fw-bold mb-5 mt-4">{result.payload.class}</h1>

              <div className="row justify-content-center w-100">
                <div className="col-4 mb-5">
                  <h2 className="mb-3">Imagen subida</h2>
                  <img src={imageURL} alt="uploaded image" className="mx-auto" style={{ maxWidth: "55%", height: "auto", display: "block" }} />
                </div>

                <div className="col-4" style={{}}>
                  <h2 className="mb-4">Accuracy</h2>
                  <GaugeChart
                    style={{ width: "110%", margin: "0 auto" }}
                    nrOfLevels={20}
                    percent={result.payload.accuracy / 100}
                    needleColor="#345243"
                    fontSize="2rem"
                    textColor="#000000"
                    colors={["#FF0000", "#00FF00"]}
                    animateDuration={3000}
                  />
                </div>
              </div>

              <Button
               className="mt-1"
                variant="primary"
                style={{ backgroundColor: "#31525B", fontSize: "2rem" }}
                onClick={() => { setDisplayResult(false); setImageUploaded(false); setResult({}); }}
              >
                Clasificar otra imagen
              </Button>
            </div>
          }
        </div>
        :
        <div>
          <h1>Adjunta una imagen de una planta</h1>
          <p className="small text-center">Recuerda que las categorías permitidas son: margaritas, rosas, dientes de león, tulipanes y girasoles</p>
          <DropZone setResult={setResult} setImageUploaded={setImageUploaded} />

          <div className="d-flex justify-content-center align-items-center mt-3">
            <Button
              variant="primary"
              style={{ backgroundColor: "#31525B", fontSize: "1.5rem" }}
              onClick={() => setDisplayResult(true)}
              disabled={!imageUploaded}
              className="me-2"
            >
              Clasificar
            </Button>
            <Instructions/>
          </div>
        </div>
      }
    </div>
  );
}

export default Home;
