import React, { useState } from "react";
import {
  Container,
  Typography,
  Box,
  Button,
  CircularProgress,
  Card,
  CardContent,
  CardMedia,
  Grid,
  Paper,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [segmentation, setSegmentation] = useState(null);
  const [visualization, setVisualization] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setSegmentation(null);
    setVisualization(null);
    setError("");
  };

  const handleSegment = async () => {
    if (!image) {
      setError("Please upload an MRI image.");
      return;
    }
    setLoading(true);
    setError("");
    setSegmentation(null);
    setVisualization(null);
    try {
      const formData = new FormData();
      formData.append("image", image);
      const response = await axios.post(
        "http://127.0.0.1:5000/segment",
        formData,
        { responseType: "blob" }
      );
      setSegmentation(URL.createObjectURL(response.data));
    } catch (err) {
      setError("Segmentation failed. Please check the backend server.");
    }
    setLoading(false);
  };

  const handleVisualize = async () => {
    if (!image) {
      setError("Please upload an MRI image.");
      return;
    }
    setLoading(true);
    setError("");
    setSegmentation(null);
    setVisualization(null);
    try {
      const formData = new FormData();
      formData.append("image", image);
      const response = await axios.post(
        "http://127.0.0.1:5000/visualize",
        formData,
        { responseType: "blob" }
      );
      setVisualization(URL.createObjectURL(response.data));
    } catch (err) {
      setError("Visualization failed. Please check the backend server.");
    }
    setLoading(false);
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%)",
        py: 4,
      }}
    >
      <Container maxWidth="md">
        <Paper elevation={6} sx={{ p: 4, borderRadius: 4 }}>
          <Typography
            variant="h3"
            align="center"
            gutterBottom
            sx={{
              fontWeight: 700,
              color: "#1976d2",
              letterSpacing: 2,
              mb: 2,
            }}
          >
            Brain MRI Segmentation
          </Typography>
          <Typography
            variant="subtitle1"
            align="center"
            gutterBottom
            sx={{ mb: 3 }}
          >
            Upload a brain MRI image and get a segmented mask or visualization overlay using AI.
          </Typography>
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              mb: 3,
            }}
          >
            <Button
              variant="contained"
              component="label"
              startIcon={<CloudUploadIcon />}
              sx={{ mb: 2, fontWeight: 600, fontSize: 16 }}
              color="primary"
            >
              Upload MRI Image
              <input
                type="file"
                accept="image/*"
                hidden
                onChange={handleImageChange}
              />
            </Button>
            {image && (
              <Card sx={{ maxWidth: 320, mb: 2 }}>
                <CardMedia
                  component="img"
                  height="200"
                  image={URL.createObjectURL(image)}
                  alt="Uploaded MRI"
                  sx={{ objectFit: "contain", background: "#f5f5f5" }}
                />
                <CardContent>
                  <Typography variant="body2" color="text.secondary">
                    {image.name}
                  </Typography>
                </CardContent>
              </Card>
            )}
            <Grid container spacing={2} justifyContent="center">
              <Grid item>
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleSegment}
                  disabled={loading || !image}
                  sx={{ fontWeight: 600 }}
                >
                  Run Segmentation
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="outlined"
                  color="success"
                  onClick={handleVisualize}
                  disabled={loading || !image}
                  sx={{ fontWeight: 600 }}
                >
                  Visualize Overlay
                </Button>
              </Grid>
            </Grid>
            {loading && <CircularProgress sx={{ mt: 3 }} />}
            {error && (
              <Typography color="error" sx={{ mt: 2 }}>
                {error}
              </Typography>
            )}
          </Box>
          {(segmentation || visualization) && (
            <Box sx={{ mt: 4 }}>
              <Typography
                variant="h5"
                align="center"
                sx={{ fontWeight: 600, color: "#1976d2", mb: 2 }}
              >
                {segmentation
                  ? "Segmentation Result"
                  : "Segmentation Visualization"}
              </Typography>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  flexDirection: "column",
                }}
              >
                {segmentation && (
                  <img
                    src={segmentation}
                    alt="Segmentation Result"
                    style={{
                      maxWidth: "100%",
                      borderRadius: 8,
                      boxShadow: "0 4px 24px rgba(0,0,0,0.1)",
                    }}
                  />
                )}
                {visualization && (
                  <img
                    src={visualization}
                    alt="Segmentation Visualization"
                    style={{
                      maxWidth: "100%",
                      borderRadius: 8,
                      boxShadow: "0 4px 24px rgba(0,0,0,0.1)",
                    }}
                  />
                )}
              </Box>
            </Box>
          )}
        </Paper>
        <Typography
          variant="body2"
          align="center"
          sx={{ mt: 4, color: "#888" }}
        >
          Built with ❤️ using React & Flask | Brain MRI AI Segmentation
        </Typography>
      </Container>
    </Box>
  );
}

export default App;
