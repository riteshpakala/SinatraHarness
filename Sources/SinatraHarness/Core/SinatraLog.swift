import Foundation

/// A tiny log sink so the package needs no logging dependency. Callers bridge it
/// to swift-log, os.Logger or print.
public protocol SinatraLog: Sendable {
    func log(_ level: SinatraLogLevel, _ message: @autoclosure () -> String)
}

public enum SinatraLogLevel: Int, Sendable, Comparable {
    case trace, debug, info, warning, error
    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// Prints to stderr at or above a threshold. Used by the CLI and tests.
public struct PrintLog: SinatraLog {
    public let threshold: SinatraLogLevel
    public init(threshold: SinatraLogLevel = .info) { self.threshold = threshold }
    public func log(_ level: SinatraLogLevel, _ message: @autoclosure () -> String) {
        guard level >= threshold else { return }
        FileHandle.standardError.write(Data("[sinatra:\(level)] \(message())\n".utf8))
    }
}
