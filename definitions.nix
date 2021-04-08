{ # `git ls-remote https://github.com/nixos/nixpkgs-channels nixos-unstable`
  nixpkgs-rev ? "266dc8c3d052f549826ba246d06787a219533b8f"
  # pin nixpkgs to the specified revision if not overridden
, pkgsPath ? builtins.fetchTarball {
    name = "nixpkgs-${nixpkgs-rev}";
    url = "https://github.com/nixos/nixpkgs/archive/${nixpkgs-rev}.tar.gz";
  }
, pkgs ? import pkgsPath {}
}: let
  lib = pkgs.lib;
in {
  inherit pkgs;
  pythonEnv = pkgs.poetry2nix.mkPoetryEnv {
    projectDir = ./.;
    python = pkgs.python38;
    overrides = pkgs.poetry2nix.overrides.withDefaults (self: super: {
      sphinx-rtd-theme = super.sphinx_rtd_theme;
      pillow = super.pillow.overridePythonAttrs (
        old: {
          # https://github.com/nix-community/poetry2nix/issues/180
          buildInputs = with pkgs; [ xorg.libX11 ] ++ old.buildInputs;
        }
      );
      matplotlib = super.matplotlib.overridePythonAttrs (
        old: {
          propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ self.certifi ];
        }
      );
      theano = self.theano-pymc;
    });
    };
}
