function ImageThumbnail({url, downloadName, maxWidth}) {
    return (
        <>
        <div className="d-inline">
                <a data-bs-toggle="tooltip" data-bs-placement="left" title={downloadName} href={url} download={downloadName}><img style={{ maxWidth, height: "auto" }} className="mx-1" src={url} /></a>
        </div>
        </>
    );
}

ImageThumbnail.defaultProps = {
    maxWidth: "18%"
}

export default ImageThumbnail;