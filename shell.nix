let
  # There is no way to pass custom arguments (such as a nixpkgs override) to
  # definitions.nix right now. There probably should be.
  inherit (import ./definitions.nix {}) pkgs pythonEnv;
in pkgs.mkShell {
  buildInputs = [
    # Unfortunately the poetry2nix setup is currently broken due to an issue
    # with newer versions of matplotlib. If you need it, you can temporarily
    # downgrade matplotlib  by executing `poetry add 'matplotlib<3.4'`. You can
    # then uncomment the `pythonEnv`. See [1] for more information.
    # [1] https://github.com/nix-community/poetry2nix/issues/280#issuecomment-815064853
    # pythonEnv
    pkgs.python38.pkgs.poetry
  ];
}
