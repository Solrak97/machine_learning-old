import ImageThumbnail from "./ImageThumbnail";

const images = [
    {
        url: "/assets/images/daisy.jpg",
        downloadName: "daisy"
    },
    {
        url: "/assets/images/rose.jpg",
        downloadName: "rose"
    },
    {
        url: "/assets/images/dandelion.jpg",
        downloadName: "dandelion"
    },
    {
        url: "/assets/images/tulip.jpg",
        downloadName: "tulip"
    },
    {
        url: "/assets/images/sunflower.jpg",
        downloadName: "sunflower"
    }
];

function Instructions() {
    return (
        <>
            <a data-bs-toggle="modal" style={{ cursor: "pointer" }} data-bs-target="#instructionsModal">
                <span data-bs-toggle="tooltip" data-bs-placement="left" title="Ayuda" data-bs-animation="true" style={{ whiteSpace: "nowrap" }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="" className="bi bi-question-circle d-block" viewBox="0 0 16 16">
                        <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
                        <path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z" />
                    </svg>
                </span>
            </a>

            <div className="modal fade" id="instructionsModal" tabIndex="-1" aria-labelledby="einstructionsLabel" aria-hidden="true">
                <div className="modal-dialog modal-dialog-centered">
                    <div className="modal-content">
                        <div className="modal-header">
                            <h5 className="modal-title fw-bold" id="instructionsLabel">Instrucciones</h5>
                            <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div className="modal-body">
                            <ol className="text-start fs-6">
                                <li> Arrastre o seleccione una imagen de la planta a clasificar.</li>
                                <li className="mt-2">En caso de no contar con imágenes, puedes descargar algunas de estas dando click en la imagen.</li>

                                <div className="my-2">
                                    {images.map(image => (
                                        <ImageThumbnail
                                            url={image.url}
                                            maxWidth={image.maxWidth}
                                            downloadName={image.downloadName}
                                            key={image.downloadName}
                                        />
                                    ))}
                                </div>
                                <li className="mt-1"> Presione el botón "Clasificar".</li>
                                <li className="mt-2"> Se mostrará la clase de la imagen y la confianza del modelo en esa clase.</li>
                            </ol>
                        </div>
                        <div className="modal-footer">
                            <button type="button" className="btn btn-primary" data-bs-dismiss="modal">Cerrar</button>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}

export default Instructions;