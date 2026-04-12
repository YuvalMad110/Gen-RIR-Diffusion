// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// Modified by Yuval Mader for Gen-RIR-Diffusion:
//   - Renders from arbitrary camera poses (CSV file) instead of a fixed trajectory
//   - Supports --rgb-only and --depth-only flags
//   - Saves output to a specified directory with caller-supplied filenames
//
// Usage:
//   ReplicaPoseRenderer <mesh.ply> <atlas_folder> <poses_csv> <output_dir>
//                       [--rgb-only|--depth-only] [glass.sur]
//
// poses_csv format (space-separated, '#' lines are comments):
//   eye_x eye_y eye_z  tgt_x tgt_y tgt_z  stem
//   e.g.: -0.665 2.036 0.970  -0.165 2.036 0.470  tgt5_src10
//
// Output:
//   <output_dir>/<stem>.jpg   (RGB, 1280x960, JPEG)
//   <output_dir>/<stem>.png   (depth, 16-bit PNG, scale = value / (65535 * 0.1) metres)

#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>

#include "GLCheck.h"
#include "MirrorRenderer.h"

#include <experimental/filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::experimental::filesystem;


int main(int argc, char* argv[]) {
  ASSERT(argc >= 5,
    "Usage: ./ReplicaPoseRenderer mesh.ply /path/to/atlases poses.csv output_dir "
    "[--rgb-only|--depth-only] [glass.sur]");

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  const std::string posesFile(argv[3]);
  const std::string outDir(argv[4]);

  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));
  ASSERT(pangolin::FileExists(posesFile));

  // Parse optional flags (args 5+)
  bool renderRgb   = true;
  bool renderDepth = true;
  std::string surfaceFile;
  for (int i = 5; i < argc; ++i) {
    std::string arg(argv[i]);
    if      (arg == "--depth-only") renderRgb   = false;
    else if (arg == "--rgb-only")   renderDepth = false;
    else                            surfaceFile  = arg;
  }

  if (surfaceFile.length()) {
    ASSERT(pangolin::FileExists(surfaceFile));
  }

  // Create output directory
  fs::create_directories(outDir);

  const int width  = 1280;
  const int height = 960;
  const float depthScale = 65535.0f * 0.1f;

  // Setup EGL
  EGLCtx egl;
  egl.PrintInformation();
  if (!checkGLVersion()) {
    return 1;
  }

  // Don't draw backfaces
  const GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup framebuffers
  pangolin::GlTexture render(width, height);
  pangolin::GlRenderBuffer renderBuffer(width, height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);

  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // Setup camera — intrinsics match original ReplicaRenderer (pinhole, 1280x960)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_BottomLeft(
          width, height,
          width  / 2.0f,
          width  / 2.0f,
          (width  - 1.0f) / 2.0f,
          (height - 1.0f) / 2.0f,
          0.1f, 100.0f),
      pangolin::ModelViewLookAtRDF(0, 0, 4, 1, 0, 4, 0, 0, 1));  // placeholder (look along +X), overwritten per pose

  // Load mirrors / glass surfaces
  std::vector<MirrorSurface> mirrors;
  if (surfaceFile.length()) {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);
    for (size_t i = 0; i < json.size(); i++) {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirror(s)" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // Load mesh and PTex texture atlases
  PTexMesh ptexMesh(meshFile, atlasFolder);

  // HDR tone-mapping — Replica textures are linear HDR; these match the ReplicaViewer defaults
  ptexMesh.SetExposure(0.025f);
  ptexMesh.SetGamma(1.6969f);

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float>    depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  // Count non-comment lines for progress display
  size_t total = 0;
  {
    std::ifstream cf(posesFile);
    std::string ln;
    while (std::getline(cf, ln))
      if (!ln.empty() && ln[0] != '#') total++;
  }

  // Main render loop — one pose per line
  std::ifstream poses(posesFile);
  ASSERT(poses.is_open(), "Cannot open poses CSV: " + posesFile);

  size_t done = 0;
  std::string line;
  while (std::getline(poses, line)) {
    if (line.empty() || line[0] == '#') continue;

    double ex, ey, ez, tx, ty, tz;
    std::string stem;
    std::istringstream iss(line);
    if (!(iss >> ex >> ey >> ez >> tx >> ty >> tz >> stem)) continue;

    done++;
    std::cout << "\rRendering " << done << "/" << total << ": " << stem << "...  ";
    std::cout.flush();

    // Set camera pose for this pair
    // up = (0, 0, 1): Pangolin's RDF LookAt uses Camera_Down = Forward × (Forward × up),
    // so passing (0,0,1) makes Camera_Up = (0,0,-1) = -Z = real-world up in Replica's Z-down mesh.
    s_cam.SetModelViewMatrix(
        pangolin::ModelViewLookAtRDF(ex, ey, ez, tx, ty, tz, 0, 0, 1));

    // ── RGB render ──────────────────────────────────────────────────────────
    if (renderRgb) {
      frameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glEnable(GL_CULL_FACE);
      ptexMesh.Render(s_cam);
      glDisable(GL_CULL_FACE);

      glPopAttrib();
      frameBuffer.Unbind();

      // Render mirrors on top
      for (size_t m = 0; m < mirrors.size(); m++) {
        MirrorSurface& mirror = mirrors[m];
        mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

        frameBuffer.Bind();
        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, width, height);
        mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(m), s_cam);
        glPopAttrib();
        frameBuffer.Unbind();
      }

      render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);
      pangolin::SaveImage(
          image.UnsafeReinterpret<uint8_t>(),
          pangolin::PixelFormatFromString("RGB24"),
          outDir + "/" + stem + ".jpg");
    }

    // ── Depth render ─────────────────────────────────────────────────────────
    if (renderDepth) {
      depthFrameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glEnable(GL_CULL_FACE);
      ptexMesh.RenderDepth(s_cam, depthScale);
      glDisable(GL_CULL_FACE);

      glPopAttrib();
      depthFrameBuffer.Unbind();

      depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);
      for (size_t j = 0; j < depthImage.Area(); j++)
        depthImageInt[j] = static_cast<uint16_t>(depthImage[j] + 0.5f);

      pangolin::SaveImage(
          depthImageInt.UnsafeReinterpret<uint8_t>(),
          pangolin::PixelFormatFromString("GRAY16LE"),
          outDir + "/" + stem + ".png", true, 34.0f);
    }
  }

  std::cout << "\nDone. Rendered " << done << " pair(s)." << std::endl;
  return 0;
}
