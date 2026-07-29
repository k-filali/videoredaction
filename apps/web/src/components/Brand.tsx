import { Icon } from "./Icon";

export function Brand() {
  return (
    <span className="brand-lockup">
      <span className="brand-symbol" aria-hidden="true">
        <span />
      </span>
      <span className="brand-copy">
        <strong>ClearFrame</strong>
        <small>Secure redaction workspace</small>
      </span>
    </span>
  );
}
export function PrivacyStatus() {
  return (
    <span className="privacy-status">
      <Icon name="shield" size={15} />
      Local processing
    </span>
  );
}
