import React, { useState } from "react";
import { Dropzone, FileItem, FullScreenPreview } from "@dropzone-ui/react";

function DropZone({setResult, setImageUploaded}) {
  const [files, setFiles] = useState([]);

  const [imageSrc, setImageSrc] = useState(undefined);

  const maxFiles = 1;

  const updateFiles = (incommingFiles) => {
    const validFiles = incommingFiles.filter((file) => file.valid).slice(-1);
    setImageUploaded(validFiles[0].file);
    setFiles(validFiles);
  };

  const onDelete = (id) => {
    setFiles(files.filter((x) => x.id !== id));
  };

  const handleSee = (imageSource) => {
    setImageSrc(imageSource);
  };

  const handleResponse = (response) => {
    const res = response.at(-1).serverResponse;    
    setResult((Object.keys(res).length === 0 || res.message === 'Error')? {error: 'Error'}: res);
  }

  return (
    <Dropzone
      style={{
        width: "20%",
        margin: "0 auto",
        fontSize: "0.7rem",
        color: "#000",
        backgroundColor: "rgba(255,255,255,0.3)",
      }}
      onUploadFinish={handleResponse}
      onChange={updateFiles}
      minHeight="200px"
      value={files}
      header={false}
      footer={false}
      label="Arrastra una imagen"
      accept=".jpg"
      url="http://localhost:5000"
      uploadOnDrop
      localization={"ES-es"}
      disableScroll
    >
      {files.map((file) => (
        <FileItem
          {...file}
          key={file.id}
          onDelete={onDelete}
          onSee={handleSee}
          localization={"ES-es"}
          preview
          hd
        />
      ))}
      <FullScreenPreview
        imgSource={imageSrc}
        openImage={imageSrc}
        onClose={(e) => handleSee(undefined)}
      />
    </Dropzone>
  );
}

export default DropZone;
