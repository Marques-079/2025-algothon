{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "price-denoise-env";

  buildInputs = with pkgs; [
    python311
    python311Packages.pip
    python311Packages.setuptools
    python311Packages.wheel
    python311Packages.numpy
    python311Packages.pandas
  ];

  shellHook = ''
    echo "🔧 Python denoising environment ready."
    echo "Use ffmpeg + scipy + matplotlib for your price signal pipeline."
  '';
}
