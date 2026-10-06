//
//  FeatureStore.swift
//  SinatraHarness
//
//  WHAT: Loads, caches and persists owner ledgers, weight models and traces under a
//        caller-supplied directory.
//
//    <root>/index.json
//    <root>/owners/<ownerKey>/ledger.json
//    <root>/owners/<ownerKey>/model-<fnv1a64(modelKey)>.safetensors
//    <root>/owners/<ownerKey>/traces/<turnId>.json
//
//  PIN:  Not thread-safe by design: exactly one `SinatraSession` actor owns a store.
//        Writes are atomic (temp file + replace). Saving is explicit (`save`/`saveDirty`)
//        so it can happen after a generation rather than in front of it.
//

import Foundation

final class FeatureStore {
    let root: URL
    private let configuration: SinatraConfiguration
    private var cache: [OwnerID: OwnerLedger] = [:]
    private var dirty: Set<OwnerID> = []
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let log: SinatraLog?

    init(root: URL, configuration: SinatraConfiguration, log: SinatraLog?) {
        self.root = root
        self.configuration = configuration
        self.log = log
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        self.encoder = encoder
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder
    }

    // MARK: Paths

    static func ownerKey(_ owner: OwnerID) -> String {
        let allowed = Set("abcdefghijklmnopqrstuvwxyz0123456789-_")
        let prefix = String(owner.rawValue.lowercased().filter { allowed.contains($0) }.prefix(32))
        let hash = FNV1a.hex(owner.rawValue)
        return prefix.isEmpty ? hash : "\(prefix)-\(hash)"
    }

    func ownerDirectory(_ owner: OwnerID) -> URL {
        root.appendingPathComponent("owners", isDirectory: true)
            .appendingPathComponent(Self.ownerKey(owner), isDirectory: true)
    }

    func ledgerURL(_ owner: OwnerID) -> URL {
        ownerDirectory(owner).appendingPathComponent("ledger.json")
    }

    func modelURL(_ owner: OwnerID, modelKey: String) -> URL {
        ownerDirectory(owner).appendingPathComponent("model-\(FNV1a.hex(modelKey)).safetensors")
    }

    func tracesDirectory(_ owner: OwnerID) -> URL {
        ownerDirectory(owner).appendingPathComponent("traces", isDirectory: true)
    }

    func traceURL(_ owner: OwnerID, turnId: UUID) -> URL {
        tracesDirectory(owner).appendingPathComponent("\(turnId.uuidString.lowercased()).json")
    }

    // MARK: Ledgers

    func ledger(_ owner: OwnerID, now: Date = Date()) -> OwnerLedger {
        if let cached = cache[owner] { return cached }
        let url = ledgerURL(owner)
        if let data = try? Data(contentsOf: url) {
            let schema = (try? decoder.decode(SchemaProbe.self, from: data))?.schema
            if let schema, schema != OwnerLedger.schema {
                log?.log(.info, "ledger schema \(schema) for \(owner) is not \(OwnerLedger.schema); starting fresh")
            } else {
                do {
                    let ledger = try decoder.decode(OwnerLedger.self, from: data)
                    cache[owner] = ledger
                    return ledger
                } catch {
                    log?.log(.warning, "ledger for \(owner) did not decode (\(error)); starting fresh")
                }
            }
        }
        let fresh = OwnerLedger(owner: owner, now: now, configuration: configuration)
        cache[owner] = fresh
        return fresh
    }

    @discardableResult
    func update<T>(_ owner: OwnerID, now: Date = Date(), _ body: (inout OwnerLedger) throws -> T) rethrows -> T {
        var ledger = self.ledger(owner, now: now)
        let result = try body(&ledger)
        ledger.updatedAt = now
        cache[owner] = ledger
        dirty.insert(owner)
        return result
    }

    func hasLedger(_ owner: OwnerID) -> Bool {
        cache[owner] != nil || FileManager.default.fileExists(atPath: ledgerURL(owner).path)
    }

    func save(_ owner: OwnerID) throws {
        guard let ledger = cache[owner] else { return }
        let data = try encoder.encode(ledger)
        try AtomicFile.write(data, to: ledgerURL(owner))
        dirty.remove(owner)
        try updateIndex(owner: owner, updatedAt: ledger.updatedAt)
    }

    func saveDirty() throws {
        for owner in dirty { try save(owner) }
    }

    func isDirty(_ owner: OwnerID) -> Bool { dirty.contains(owner) }

    func forget(_ owner: OwnerID) throws {
        cache[owner] = nil
        dirty.remove(owner)
        let directory = ownerDirectory(owner)
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
        try updateIndex(owner: owner, updatedAt: nil)
    }

    // MARK: Traces

    func saveTrace(_ trace: InjectionTrace, owner: OwnerID) throws {
        let data = try encoder.encode(trace)
        try AtomicFile.write(data, to: traceURL(owner, turnId: trace.traceId))
        try pruneTraces(owner)
    }

    func loadTrace(owner: OwnerID, turnId: UUID) -> InjectionTrace? {
        guard let data = try? Data(contentsOf: traceURL(owner, turnId: turnId)) else { return nil }
        return try? decoder.decode(InjectionTrace.self, from: data)
    }

    private func pruneTraces(_ owner: OwnerID) throws {
        let directory = tracesDirectory(owner)
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: keys)
        else { return }
        let traces = files.filter { $0.pathExtension == "json" }
        guard traces.count > configuration.traceHistory else { return }
        let dated = traces.map { url -> (URL, Date) in
            let date = (try? url.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
            return (url, date)
        }.sorted { $0.1 < $1.1 }
        for (url, _) in dated.prefix(traces.count - configuration.traceHistory) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private struct SchemaProbe: Decodable {
        var schema: Int
    }

    // MARK: Index

    private struct Index: Codable {
        var schema = 1
        var owners: [Entry] = []
        struct Entry: Codable {
            var key: String
            var owner: String
            var updatedAt: Date
        }
    }

    private func updateIndex(owner: OwnerID, updatedAt: Date?) throws {
        let url = root.appendingPathComponent("index.json")
        var index = (try? Data(contentsOf: url)).flatMap { try? decoder.decode(Index.self, from: $0) } ?? Index()
        index.owners.removeAll { $0.owner == owner.rawValue }
        if let updatedAt {
            index.owners.append(.init(key: Self.ownerKey(owner), owner: owner.rawValue, updatedAt: updatedAt))
        }
        index.owners.sort { $0.owner < $1.owner }
        try AtomicFile.write(try encoder.encode(index), to: url)
    }

    func knownOwners() -> [OwnerID] {
        let url = root.appendingPathComponent("index.json")
        guard let data = try? Data(contentsOf: url), let index = try? decoder.decode(Index.self, from: data) else {
            return Array(cache.keys)
        }
        return index.owners.map { OwnerID($0.owner) }
    }
}

enum AtomicFile {
    static func write(_ data: Data, to url: URL) throws {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let temporary = directory.appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        try data.write(to: temporary)
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: url)
        }
    }
}
