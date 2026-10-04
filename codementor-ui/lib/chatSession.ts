/**
 * Random id for one chat conversation, sent with each question so the server can keep its last
 * few turns. It lives only in the chat window's state: reloading the page starts a new chat on
 * screen and a new id, so the model never uses turns the user can no longer see. The server
 * accepts ids matching /^[A-Za-z0-9_-]{16,128}$/.
 */
export function newChatSessionId(cryptoImpl: Pick<Crypto, "getRandomValues"> & Partial<Pick<Crypto, "randomUUID">> = globalThis.crypto): string {
    if (typeof cryptoImpl.randomUUID === "function") return cryptoImpl.randomUUID();
    // randomUUID needs a secure context (https or localhost); getRandomValues works everywhere.
    const bytes = cryptoImpl.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
