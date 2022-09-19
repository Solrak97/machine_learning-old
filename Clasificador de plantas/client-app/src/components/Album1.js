import React, { useState, useCallback } from "react";
import adv1 from "../pictures/album1/adv1.jpg";
import adv2 from "../pictures/album1/adv2.jpg";
import adv3 from "../pictures/album1/adv3.jpg";
import adv5 from "../pictures/album1/adv5.jpg";
import adv7 from "../pictures/album1/adv7.jpg";
import ImageViewer from "react-simple-image-viewer";
import "../Album.css";

function Album2() {
  const [currentImage, setCurrentImage] = useState(0);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const images = [adv1, adv2, adv3, adv5, adv7];

  const openImageViewer = useCallback((index) => {
    setCurrentImage(index);
    setIsViewerOpen(true);
  }, []);

  const closeImageViewer = () => {
    setCurrentImage(0);
    setIsViewerOpen(false);
  };

  return (
    <div className="row">
      {images.map((src, index) => (
        <div className="collumn">
          <img
            src={src}
            onClick={() => openImageViewer(index)}
            width="auto"
            key={index}
            style={{ margin: "2px" }}
            alt=""
          />
        </div>
      ))}

      {isViewerOpen && (
        <ImageViewer
          src={images}
          currentIndex={currentImage}
          onClose={closeImageViewer}
          disableScroll={true}
          backgroundStyle={{
            backgroundColor: "rgba(0,0,0,0.9)"
          }}
        />
      )}
    </div>
  );
}

export default Album2;
