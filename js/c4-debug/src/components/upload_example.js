import React from "react";

/**
 * Component to handle file upload. Works for image
 * uploads, but can be edited to work for any file.
 */
function FileUpload() {
  // State to store uploaded file
  const [file, setFile] = React.useState("");

  console.log(typeof(file));
  console.log(typeof(file.size));
  // Handles file upload event and updates state
  function handleUpload(event) {
    setFile(event.target.files[0]);

    // Add code here to upload file to server
    // ...
  }

  return (
    <div id="upload-box">
      <input type="file" onChange={handleUpload} />
      <p>Filename: {file.name}</p>
      <p>File type: {file.type}</p>
      <p>File size: {file.size} bytes</p>
      {file && <ImageThumb image={file} />}
    </div>
  );
}

/**
 * Component to display thumbnail of image.
 */
const ImageThumb = ({ image }) => {
  return <img src={URL.createObjectURL(image)} alt={image.name} />;
};


export default function App() {
  return <FileUpload />;
}