import crypto from "crypto";

export function sessionSeed(sessionId: string): number {
  const hash = crypto.createHash("sha256").update(sessionId).digest();
  return hash.readUInt32BE(0);
}

export function createCaptchaCode(): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

  let code = "";
  for (let i = 0; i < 5; i++) {
    const idx = crypto.randomInt(0, alphabet.length);
    code += alphabet[idx];
  }
  return code;
}
