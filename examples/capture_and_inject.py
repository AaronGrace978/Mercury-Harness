from mercury import MercuryHarness
from mercury.demo import frontier_auth_fix, lesser_auth_fail


def main() -> None:
    harness = MercuryHarness.init(".mercury")
    harness.capture(frontier_auth_fix())
    harness.contrast(lesser_auth_fail(), frontier_auth_fix())
    pack = harness.pack(
        "Login redirects back to /login after a successful password check",
        model="gpt-4o-mini",
    )
    print(pack.render())


if __name__ == "__main__":
    main()
