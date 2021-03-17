{ # `git ls-remote https://github.com/nixos/nixpkgs-channels nixos-unstable`
  nixpkgs-rev ? "266dc8c3d052f549826ba246d06787a219533b8f"
  # pin nixpkgs to the specified revision if not overridden
, pkgsPath ? builtins.fetchTarball {
    name = "nixpkgs-${nixpkgs-rev}";
    url = "https://github.com/nixos/nixpkgs/archive/${nixpkgs-rev}.tar.gz";
  }
, pkgs ? import pkgsPath {}
}:
let
  pythonEnv = pkgs.poetry2nix.mkPoetryEnv { 
    projectDir = ./.;
    # For tensorflow 1.15 https://github.com/nix-community/poetry2nix/issues/180
    python = pkgs.python37;
    overrides = pkgs.poetry2nix.overrides.withDefaults (self: super: {
      sphinx-rtd-theme = super.sphinx_rtd_theme;
      pillow = super.pillow.overridePythonAttrs (
        old: {
          # https://github.com/nix-community/poetry2nix/issues/180
          buildInputs = with pkgs; [ xorg.libX11 ] ++ old.buildInputs;
        }
      );
    });
  };
in pkgs.mkShell {
  buildInputs = [
    # Unfortunately the poetry2nix setup is currently broken due to an issue
    # with newer versions of matplotlib. If you need it, you can temporarily
    # downgrade matplotlib  by executing `poetry add 'matplotlib<3.4'`. You can
    # then uncomment the `pythonEnv`. See [1] for more information.
    # [1] https://github.com/nix-community/poetry2nix/issues/280#issuecomment-815064853
    # pythonEnv
    pkgs.python37.pkgs.poetry
  ];
}
